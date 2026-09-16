"""
Summary chunks for lines, districts and the whole network.

Per-station chunks cannot answer aggregation questions ("how many elevated stations are on the Blue line?")
because only the top-k stations are retrieved. Pre-computed summary chunks give the LLM exact counts and full lists.
"""
import pandas as pd


def summarize_line(line_df):
    first, last = line_df.iloc[0], line_df.iloc[-1]
    type_parts = [
        f"{type_en} ({type_ar}): {len(group)} - {', '.join(group['metro_station_desc_en'])}"
        for (type_en, type_ar), group in line_df.groupby(["metro_station_type_desc_en", "metro_station_type_desc_ar"])
    ]
    type_counts_ar = "، ".join(f"{type_ar}: {n}" for type_ar, n in line_df["metro_station_type_desc_ar"].value_counts().items())
    return (
        f"Line summary: {first['metro_line_desc_en']} ({first['metro_line_desc_ar']}, {first['metro_line_cd']}) "
        f"has {len(line_df)} stations, from {first['metro_station_desc_en']} to {last['metro_station_desc_en']}. "
        f"Station types: {'; '.join(type_parts)}. "
        f"ملخص {first['metro_line_desc_ar']}: يضم {len(line_df)} محطة ({type_counts_ar})."
    )


def summarize_line_route(line_df):
    # Kept separate from the line summary so long station lists don't dilute the counts chunk
    first = line_df.iloc[0]
    districts = line_df.loc[line_df["district_match"] == "within", "district_en"].unique()
    ordered = ", ".join(
        f"{pos}. {en} ({ar})"
        for pos, en, ar in zip(line_df["position_on_line"], line_df["metro_station_desc_en"], line_df["metro_station_desc_ar"])
    )
    return (
        f"Line route: {first['metro_line_desc_en']} ({first['metro_line_desc_ar']}, {first['metro_line_cd']}) "
        f"stations in order: {ordered}. "
        f"Districts served: {', '.join(districts)}."
    )


def summarize_district(district_df):
    first = district_df.iloc[0]
    stations = (district_df
                .groupby(["metro_station_cd", "metro_station_desc_en", "metro_station_desc_ar"], sort=False)["metro_line_desc_en"]
                .apply(", ".join))
    listing = "; ".join(f"{en} ({ar}) on {lines}" for (_, en, ar), lines in stations.items())
    return (
        f"District summary: {first['district_en']} district ({first['district_ar']}) has "
        f"{len(stations)} metro station(s): {listing}. "
        f"Lines serving the district: {', '.join(district_df['metro_line_desc_en'].unique())}. "
        f"محطات المترو في {first['district_ar']}: {'، '.join(ar for _, _, ar in stations.index)}."
    )


def summarize_network(df):
    unique_stations = df.drop_duplicates("metro_station_cd")
    per_line = "; ".join(
        f"{group.iloc[0]['metro_line_desc_en']} ({group.iloc[0]['metro_line_desc_ar']}): {len(group)} stations"
        for _, group in df.groupby("metro_line_cd")
    )
    # Types are counted per line stop: an interchange can differ by line (Ministry of Education is At Grade on one line, Deep Underground on another)
    per_type = "; ".join(f"{t}: {n}" for t, n in df["metro_station_type_desc_en"].value_counts().items())
    interchanges = "; ".join(
        f"{name} ({', '.join([line] + others)})"
        for name, line, others in unique_stations.loc[unique_stations["other_lines_en"].str.len() > 0,
                                                      ["metro_station_desc_en", "metro_line_desc_en", "other_lines_en"]].values
    )
    return (
        f"Network overview: the Riyadh Metro (مترو الرياض) has {df['metro_line_cd'].nunique()} lines and "
        f"{len(unique_stations)} unique stations ({len(df)} line stops, counting interchange stations once per line). "
        f"نظرة عامة على شبكة مترو الرياض: {df['metro_line_cd'].nunique()} مسارات و{len(unique_stations)} محطة. "
        f"Stations per line: {per_line}. "
        f"Line stops by station type: {per_type}. "
        f"Stations are spread across {df.loc[df['district_match'] == 'within', 'district_en'].nunique()} districts. "
        f"Interchange stations: {interchanges}."
    )


def build_knowledge_base(df):
    """One chunk per station stop + one per line + one per line route + one per district + one network overview."""
    return pd.DataFrame(
        [{"chunk_id": f"station:{line}:{code}", "chunk_type": "station", "text_chunk": text}
         for line, code, text in zip(df["metro_line_cd"], df["metro_station_cd"], df["text_chunk"])]
        + [{"chunk_id": f"line:{line}", "chunk_type": "line", "text_chunk": summarize_line(group)}
           for line, group in df.groupby("metro_line_cd")]
        + [{"chunk_id": f"line_route:{line}", "chunk_type": "line_route", "text_chunk": summarize_line_route(group)}
           for line, group in df.groupby("metro_line_cd")]
        + [{"chunk_id": f"district:{district}", "chunk_type": "district", "text_chunk": summarize_district(group)}
           for district, group in df[df["district_match"] == "within"].groupby("district_en")]
        + [{"chunk_id": "network", "chunk_type": "network", "text_chunk": summarize_network(df)}]
    )
