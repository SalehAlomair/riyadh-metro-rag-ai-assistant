"""
Evaluate the Riyadh Metro RAG system.

Retrieval evaluation (no LLM): each question lists the chunk(s) that contain the answer. We report top-1
accuracy and top-k recall across fact, neighbor, district, aggregation and former-name questions in English and Arabic.

Usage:
    python evaluate.py              # retrieval evaluation only
    python evaluate.py --with-llm   # also run end-to-end questions through Bedrock
"""
import argparse

import pandas as pd

from metro_rag.config import TOP_K
from metro_rag.pipeline import MetroRAG


def station_ids(df, name_en, line_cd=None):
    rows = df[df["metro_station_desc_en"] == name_en]
    if line_cd:
        rows = rows[rows["metro_line_cd"] == line_cd]
    return [f"station:{line}:{code}" for line, code in zip(rows["metro_line_cd"], rows["metro_station_cd"])]


def district_ids_of(df, name_en):
    # District summaries list their stations, so they also answer "which district is X in?"
    rows = df[(df["metro_station_desc_en"] == name_en) & (df["district_match"] == "within")]
    return [f"district:{district}" for district in rows["district_en"].unique()]


def build_retrieval_tests(df):
    # (category, language, question, acceptable chunk ids)
    return [
        ("Fact", "EN", "What type of station is Dr Sulaiman Al Habib, and which line is it on?", station_ids(df, "Dr Sulaiman Al Habib")),
        ("Fact", "AR", "ما نوع محطة المتحف الوطني؟", station_ids(df, "National Museum")),
        ("Neighbor", "EN", "What is the station immediately following KAFD on the Blue Line?", station_ids(df, "KAFD", "Line1")),
        ("Neighbor", "AR", "ما هي المحطة التي تلي محطة المروج في المسار الأزرق؟", station_ids(df, "Al Murooj", "Line1")),
        ("District", "EN", "Which district is Al Murooj station located in?", station_ids(df, "Al Murooj") + district_ids_of(df, "Al Murooj")),
        ("District", "EN", "Which metro stations are in Al Olaya district?", ["district:Al Olaya"]),
        ("District", "AR", "ما هي محطات المترو الموجودة في حي الورود؟", ["district:Al Woroud"]),
        ("District", "AR", "في أي حي تقع محطة قصر الحكم؟", station_ids(df, "Qasr Al Hokm") + district_ids_of(df, "Qasr Al Hokm")),
        ("Aggregation", "EN", "How many elevated stations are there in total on the Blue Line?", ["line:Line1"]),
        ("Aggregation", "AR", "كم عدد المحطات في المسار الأحمر؟", ["line:Line2"]),
        ("Aggregation", "EN", "How many lines and stations does the Riyadh Metro network have?", ["network"]),
        ("Former name", "EN", "Which line serves the Terminal 5 station?", station_ids(df, "Airport T5")),
    ]


def evaluate_retrieval(rag, k=TOP_K):
    report_rows = []
    for category, lang, question, expected in build_retrieval_tests(rag.df):
        results = rag.retrieve(question, k=k)
        hit_rank = next((res["rank"] for res in results if res["chunk_id"] in expected), None)
        report_rows.append({
            "category": category,
            "lang": lang,
            "question": question,
            "top_1_chunk": results[0]["chunk_id"],
            "hit_rank": hit_rank,
        })

    retrieval_report = pd.DataFrame(report_rows)
    retrieval_report["top_1_hit"] = retrieval_report["hit_rank"] == 1
    retrieval_report[f"top_{k}_hit"] = retrieval_report["hit_rank"].notna()

    print(f"Top-1 accuracy: {retrieval_report['top_1_hit'].mean():.0%} | "
          f"Top-{k} recall: {retrieval_report[f'top_{k}_hit'].mean():.0%} "
          f"({len(retrieval_report)} questions)")
    with pd.option_context("display.max_columns", None, "display.max_colwidth", 60, "display.width", 250):
        print(retrieval_report)
    return retrieval_report


TEST_QUERIES = [
    "What type of station is Dr Sulaiman Al Habib, and which line is it on?",  # Fact Retrieval
    "What is the station immediately following KAFD on the Blue Line?",        # Relational Logic
    "How many elevated stations are there in total on the Blue Line?",         # Aggregation (Stress Test)
    "Which metro stations are in Al Olaya district?",                          # District lookup
    "في أي حي تقع محطة قصر الحكم؟",                                              # District (Arabic)
    "Which district is the Airport T5 station in?",                            # Outside district boundaries
]


def evaluate_end_to_end(rag, k=TOP_K):
    print("SYSTEM EVALUATION (WITH LLM GENERATION)\n" + "=" * 40)
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nTest Q{i}: '{query}'")

        # 1. Print the retrieved chunks so you can check retrieval quality
        raw_results = rag.retrieve(query, k=k)
        print("  [Retrieved Contexts]:")
        for res in raw_results:
            print(f"    -> [Rank {res['rank']} | Score: {res['score']}] {res['chunk_id']}")

        # 2. Generate the grounded answer with Bedrock, reusing the retrieved context
        print("  [LLM Response]:")
        llm_response = rag.ask(query, retrieved=raw_results)
        print(f"    🤖 Claude: {llm_response}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Riyadh Metro RAG system.")
    parser.add_argument("--with-llm", action="store_true", help="also run end-to-end questions through Amazon Bedrock")
    parser.add_argument("-k", type=int, default=TOP_K, help=f"number of chunks to retrieve (default {TOP_K})")
    args = parser.parse_args()

    rag = MetroRAG()
    print(rag.knowledge_base["chunk_type"].value_counts(), "\n")

    evaluate_retrieval(rag, k=args.k)
    if args.with_llm:
        print()
        evaluate_end_to_end(rag, k=args.k)


if __name__ == "__main__":
    main()
