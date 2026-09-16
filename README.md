# Riyadh Metro RAG Assistant

Tags: Python, Retrieval-Augmented Generation, FAISS, SentenceTransformers, Amazon Bedrock, Gradio, GeoPandas, Geospatial Enrichment, Multilingual NLP, Arabic NLP, Enterprise AI, Public Transport Analytics

## Business Value and Problem Statement

Public transport networks generate high-volume, repetitive information requests about station names, routes, station types, line assignments, districts, and nearby stops. In a city-scale metro system, manual lookup processes increase response time, create inconsistencies across channels, and introduce operational risk when staff or applications provide answers from outdated or incomplete sources.

The Riyadh Metro RAG Assistant addresses this problem by converting structured station data, enriched with the Riyadh district each station lies in, into a bilingual retrieval system. Instead of allowing a language model to answer from memory, the system retrieves station-specific evidence from a FAISS vector index and instructs the language model to answer only from that context.

This approach provides measurable business value in four areas:

- Reduces manual station lookup effort for support, mobility applications, and internal teams.
- Improves factual consistency by grounding every generated answer in structured data.
- Supports Arabic and English users through a multilingual embedding model and bilingual response policy.
- Creates an auditable architecture where retrieved context can be reviewed during testing and production monitoring.

## System Architecture and Workflow

```text
Raw Riyadh Metro station data            Riyadh district polygons (GeoJSON)
        |                                          |
        v                                          |
JSON loader and flattening                         |
        |                                          |
        v                                          v
Spatial join: station -> district (point-in-polygon, nearest-district fallback,
bordering districts within 150 m)
        |
        v
Enrichment: position on line, neighbors, interchanges, former names
        |
        v
Bilingual knowledge base
  - one chunk per station stop (94)
  - line summaries with type counts (6) and line routes (6)
  - district summaries (51)
  - network overview (1)
        |
        v
Multilingual E5 embedding model
        |
        v
Normalized vector embeddings
        |
        v
FAISS inner-product index
        |
        v
Top-k retrieved context (k = 5)
        |
        v
Grounded prompt construction
        |
        v
Amazon Bedrock language model
        |
        v
Arabic or English answer with retrieved source context
        |
        v
Gradio demonstration interface
```

## Repository Structure

```text
.
├── README.md
├── app.py                  # Gradio interface (entry point)
├── evaluate.py             # retrieval and end-to-end evaluation (entry point)
├── metro_rag/
│   ├── config.py           # paths, spatial join settings, model ids
│   ├── data_processing.py  # load stations, district join, line context, station chunks
│   ├── knowledge_base.py   # line, route, district and network summary chunks
│   ├── retrieval.py        # E5 embeddings, FAISS index, top-k retrieval
│   ├── llm.py              # Bedrock client, prompt and answer generation
│   └── pipeline.py         # MetroRAG: builds everything once, exposes retrieve() and ask()
├── metro_rag.ipynb         # original exploratory notebook
├── requirements.txt
└── data/
    ├── metro-stations-in-riyadh-by-metro-line-and-station-type-2024.json
    └── riyadh-districts.geojson   # Riyadh subset of the districts dataset, created on first run
```

## Data Sources

- **Metro stations:** `metro-stations-in-riyadh-by-metro-line-and-station-type-2024.json` (6 lines, 83 unique stations, 94 station-line stops, with coordinates).
- **Districts:** [homaily/Saudi-Arabia-Regions-Cities-and-Districts](https://github.com/homaily/Saudi-Arabia-Regions-Cities-and-Districts) district polygons, filtered to Riyadh (`city_id = 3`, 189 districts). The notebook downloads the file once and caches the Riyadh subset in `data/`.

Stations are matched to districts with a point-in-polygon join in a metric projection (UTM 38N). 89 of 94 stops fall inside a district polygon; the 5 remaining stops (PNU and airport terminals) are assigned the nearest district with the distance recorded, and their chunks state that they lie outside district boundaries. Many central stations sit on boundary roads, so districts within 150 m are also listed as bordering districts.

## Technical Stack

- Python 3.9+ (3.11 recommended)
- pandas and NumPy for data processing
- GeoPandas and Shapely for the station-to-district spatial join
- SentenceTransformers for multilingual embeddings
- FAISS for vector similarity search
- Amazon Bedrock Runtime through boto3 for generation
- Gradio for the demonstration user interface

## Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/SalehAlomair/riyadh-metro-rag-ai-assistant
cd riyadh-metro-rag-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure AWS credentials

Amazon Bedrock requires valid AWS credentials, the correct region, and access to the selected model.

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Credentials can be configured through one of the following production-safe methods:

- IAM role attached to the runtime environment.
- AWS CLI profile through `aws configure`.
- Environment variables injected by a secrets manager or CI/CD platform.

Do not commit AWS access keys to the repository.

### 5. Run the app

```bash
python app.py                  # Gradio interface
python evaluate.py             # retrieval evaluation (no AWS needed)
python evaluate.py --with-llm  # also run end-to-end questions through Bedrock
```

The notebook (`jupyter lab metro_rag.ipynb`) is kept for exploration. The first run needs internet access to download the embedding model and the districts GeoJSON; later runs use the local copies.

## Key Results and Robustness

The notebook includes a retrieval evaluation suite (section 3.1) with 12 English and Arabic questions, each labeled with the chunk(s) that contain the answer, covering:

- Direct fact retrieval: station type and line assignment.
- Relational retrieval: previous and next station questions.
- District retrieval: which district a station is in, and which stations a district contains.
- Aggregation: station counts by line, station type, and for the whole network.
- Former station names (e.g. "Terminal 5" for Airport T5).

Latest run (multilingual-e5-small, k = 5):

| Metric | Result |
| --- | --- |
| Top-1 retrieval accuracy | 92% (11/12) |
| Top-5 retrieval recall | 100% (12/12) |

The suite is small and was used while designing the chunks, so treat these numbers as a regression check rather than a held-out benchmark. Section 3.2 runs six questions end to end through Bedrock, including an Arabic district question and a station outside district boundaries.

Summary chunks fixed a failure of the earlier station-only index: "How many elevated stations are there on the Blue Line?" used to retrieve 3 arbitrary stations and answer "3"; it now retrieves the Blue line summary and answers 9.

Recommended metrics for production validation:

- Top-1 retrieval accuracy for direct station facts.
- Top-3 retrieval recall for relational questions.
- Exact-match accuracy for deterministic station counts.
- Answer faithfulness rate based on retrieved context.
- Median and p95 latency for retrieval and generation.
- Unsupported-question rejection rate.

## Edge Case Handling

The implementation is designed to handle the following operational edge cases:

- Empty user questions are rejected before retrieval.
- Stations outside every district polygon fall back to the nearest district and are labeled as such.
- Interchange stations (same station code on several lines) list the other lines they serve.
- AWS or Bedrock failures return a controlled service-availability message in the UI, with details logged instead of shown to users.
- Unknown answers are constrained by the prompt to avoid unsupported generation.
- Arabic and English queries are routed through the same multilingual retrieval pipeline.

## Security and Governance

Production deployments should follow these controls:

- Keep AWS credentials in IAM roles or a secrets manager.
- Log retrieved context and model responses for audit review.
- Version the source dataset and index artifacts.
- Add automated tests for schema validation and retrieval quality.
- Monitor unsupported questions and retrieval failures.
- Avoid exposing internal AWS errors directly to public users.

## Future Enhancements

- Persist FAISS index artifacts to disk for faster startup.
- Add schema validation for the station and district datasets.
- Replace the open-source district polygons with official municipal boundaries when available.
- Add cross-encoder reranking for higher precision on ambiguous station names.
- Add hybrid search combining keyword filters and vector retrieval.
- Implement deterministic handlers for analytics questions before LLM generation.
- Add an API layer with authentication and rate limiting.
- Add CI checks for notebook execution, linting, and unit tests.

## License

Add the appropriate license for the dataset and project code before publishing the repository.
