"""Load the station data, enrich it with districts and line context, and turn each station into a text chunk."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from .config import (
    DISTRICTS_PATH,
    DISTRICTS_URL,
    METRIC_CRS,
    NEARBY_RADIUS_M,
    RIYADH_CITY_ID,
    STATIONS_PATH,
)


def load_stations(file_path=STATIONS_PATH):
    """Load the raw JSON list and flatten nested fields (e.g. geo_point_2d.lon)."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a list of stations in {file_path}, got {type(raw_data).__name__}")

    return pd.json_normalize(raw_data)


def load_riyadh_districts(path=DISTRICTS_PATH):
    """Load Riyadh districts (download once from GitHub, then reuse the local copy)."""
    if not path.exists():
        response = requests.get(DISTRICTS_URL, timeout=60)
        response.raise_for_status()
        all_districts = gpd.GeoDataFrame.from_features(response.json()["features"], crs="EPSG:4326")
        path.parent.mkdir(parents=True, exist_ok=True)
        all_districts[all_districts["city_id"] == RIYADH_CITY_ID].to_file(path, driver="GeoJSON")

    return gpd.read_file(path)[["district_id", "name_ar", "name_en", "geometry"]]


def _strip_dist_suffix(names):
    return names.str.replace(r"\s*Dist\.$", "", regex=True)


def enrich_with_districts(df, riyadh_districts):
    """
    Match each station to the district polygon it lies in. Stations outside every polygon
    (e.g. the airport terminals) fall back to the nearest district, with the distance recorded.
    """
    df = df.copy()

    # 1) Stations as points, projected to a metric CRS so nearest-district distances are correct
    stations_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["geo_point_2d.lon"], df["geo_point_2d.lat"]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)
    districts_gdf = riyadh_districts.to_crs(METRIC_CRS)

    # 2) Point-in-polygon join, with the nearest district as fallback (keep one match per station)
    within = gpd.sjoin(stations_gdf, districts_gdf, how="left", predicate="within")
    within = within[~within.index.duplicated(keep="first")]
    nearest = gpd.sjoin_nearest(stations_gdf, districts_gdf, how="left", distance_col="distance_m")
    nearest = nearest[~nearest.index.duplicated(keep="first")]

    is_within = within["name_ar"].notna()
    df["district_ar"] = within["name_ar"].where(is_within, nearest["name_ar"])
    df["district_en"] = _strip_dist_suffix(within["name_en"].where(is_within, nearest["name_en"]))
    df["district_match"] = np.where(is_within, "within", "nearest")
    df["district_distance_m"] = np.where(is_within, 0, nearest["distance_m"].round()).astype(int)

    # 3) Many stations sit on boundary roads, so also record other districts within a short radius
    touching = gpd.sjoin(
        stations_gdf[["geometry"]].set_geometry(stations_gdf.buffer(NEARBY_RADIUS_M)),
        districts_gdf,
        how="inner",
        predicate="intersects",
    )
    touching["label"] = _strip_dist_suffix(touching["name_en"]) + " (" + touching["name_ar"] + ")"
    touching_labels = touching.groupby(level=0)["label"].apply(list)
    df["nearby_districts"] = [
        [label for label in touching_labels.get(i, []) if not label.startswith(f"{own} (")]
        for i, own in zip(df.index, df["district_en"])
    ]

    return df


def add_line_context(df):
    """Add position on line, previous/next stations, interchanges and former names."""
    # 1. metro_station_seq runs across the whole network (the Red line starts at 26),
    # so we also derive each station's position within its own line
    df = df.sort_values(by=["metro_line_cd", "metro_station_seq"]).reset_index(drop=True)
    df["position_on_line"] = df.groupby("metro_line_cd").cumcount() + 1
    df["stations_on_line"] = df.groupby("metro_line_cd")["metro_station_cd"].transform("count")

    # 2. Previous and next stations within the same metro line
    for lang in ["en", "ar"]:
        line_stations = df.groupby("metro_line_cd")[f"metro_station_desc_{lang}"]
        df[f"prev_station_{lang}"] = line_stations.shift(1)
        df[f"next_station_{lang}"] = line_stations.shift(-1)

    # 3. Interchanges: the same station code appears on more than one line
    lines_by_station = df.groupby("metro_station_cd")["metro_line_desc_en"].apply(list)
    df["other_lines_en"] = [
        [line for line in lines_by_station[code] if line != own_line]
        for code, own_line in zip(df["metro_station_cd"], df["metro_line_desc_en"])
    ]

    # 4. Former English names recorded in the comments (e.g. "Terminal 5" -> "Airport T5")
    df["former_name_en"] = df["comments_en"].str.extract(r'changed from "([^"]+)"', expand=False)

    return df


def textify_metro(row):
    """Dense key-value style chunk for one station stop, to eliminate token bloat."""
    chunk = (
        f"Station: {row['metro_station_desc_en']} ({row['metro_station_desc_ar']}). "
        f"Line: {row['metro_line_desc_en']} ({row['metro_line_desc_ar']}, {row['metro_line_cd']}). "
        f"Type: {row['metro_station_type_desc_en']} ({row['metro_station_type_desc_ar']}). "
        f"Position: station {row['position_on_line']} of {row['stations_on_line']} on the {row['metro_line_desc_en']}. "
    )

    if row["district_match"] == "within":
        chunk += f"District: {row['district_en']} ({row['district_ar']}). "
    else:
        chunk += (
            f"District: outside district boundaries; nearest district is {row['district_en']} "
            f"({row['district_ar']}), about {row['district_distance_m'] / 1000:.1f} km away. "
        )
    if row["nearby_districts"]:
        chunk += f"Bordering districts (within {NEARBY_RADIUS_M} m): {', '.join(row['nearby_districts'])}. "

    # Relational context if neighbors exist (ends of the line are marked as terminus)
    if pd.notna(row["prev_station_en"]):
        chunk += f"Previous station: {row['prev_station_en']} ({row['prev_station_ar']}). "
    if pd.notna(row["next_station_en"]):
        chunk += f"Next station: {row['next_station_en']} ({row['next_station_ar']}). "
    if row["position_on_line"] in (1, row["stations_on_line"]):
        chunk += f"Terminus of the {row['metro_line_desc_en']}. "

    if row["other_lines_en"]:
        chunk += f"Interchange: also served by {', '.join(row['other_lines_en'])}. "
    if pd.notna(row["former_name_en"]):
        chunk += f"Formerly named: {row['former_name_en']}. "

    return chunk.strip()


def prepare_stations():
    """Full preprocessing pipeline: load, enrich with districts and line context, and textify."""
    df = load_stations()
    df = enrich_with_districts(df, load_riyadh_districts())
    df = add_line_context(df)
    df["text_chunk"] = [textify_metro(row) for _, row in df.iterrows()]
    return df
