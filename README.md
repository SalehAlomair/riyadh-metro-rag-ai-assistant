# Riyadh Metro RAG Assistant

Tags: Python, Retrieval-Augmented Generation, FAISS, SentenceTransformers, Amazon Bedrock, Gradio, Multilingual NLP, Arabic NLP, Enterprise AI, Public Transport Analytics

## Business Value and Problem Statement

Public transport networks generate high-volume, repetitive information requests about station names, routes, station types, line assignments, and nearby stops. In a city-scale metro system, manual lookup processes increase response time, create inconsistencies across channels, and introduce operational risk when staff or applications provide answers from outdated or incomplete sources.

The Riyadh Metro RAG Assistant addresses this problem by converting structured station data into a bilingual retrieval system. Instead of allowing a language model to answer from memory, the system retrieves station-specific evidence from a FAISS vector index and instructs the language model to answer only from that context.

This approach provides measurable business value in four areas:

- Reduces manual station lookup effort for support, mobility applications, and internal teams.
- Improves factual consistency by grounding every generated answer in structured data.
- Supports Arabic and English users through a multilingual embedding model and bilingual response policy.
- Creates an auditable architecture where retrieved context can be reviewed during testing and production monitoring.

## System Architecture and Workflow

```text
Raw Riyadh Metro station data
        |
        v
Dataset loader and schema validation
        |
        v
Preprocessing and station sequence enrichment
        |
        v
Bilingual station text chunks
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
Top-k retrieved station context
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
├── riyadh_metro_rag_enterprise.ipynb
├── requirements.txt
├── Dockerfile
├── .env.example
└── data/
    └── metro-stations-in-riyadh-by-metro-line-and-station-type-2024.json
```

The dataset file is expected in the project root by default. For a cleaner repository layout, place it under `data/` and update `CONFIG.dataset_path` in the notebook.

## Technical Stack

- Python 3.11
- pandas and NumPy for data processing
- SentenceTransformers for multilingual embeddings
- FAISS for vector similarity search
- Amazon Bedrock Runtime through boto3 for generation
- Gradio for the demonstration user interface
- Docker for reproducible deployment

## Installation Guide

### 1. Clone the repository

```bash
git clone <repository-url>
cd riyadh-metro-rag-assistant
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

### 5. Run the notebook

```bash
jupyter lab riyadh_metro_rag_enterprise.ipynb
```

Run the cells sequentially after placing the dataset file in the configured path.

## Docker Deployment

Build the container:

```bash
docker build -t riyadh-metro-rag .
```

Run the container with local AWS credentials mounted:

```bash
docker run --rm -p 7860:7860 \
  -e AWS_REGION=us-east-1 \
  -e BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
  -v ~/.aws:/root/.aws:ro \
  -v $(pwd)/data:/app/data:ro \
  riyadh-metro-rag
```

For GPU-enabled environments, use an NVIDIA container runtime and install compatible CUDA-enabled PyTorch and FAISS builds. The CPU configuration is sufficient for small to medium station datasets, while GPU acceleration is recommended for large-scale document expansion, batch embedding jobs, or frequent index rebuilds.

## GPU Acceleration Notes

The main GPU benefit in this project is embedding generation. FAISS CPU search is already efficient for a compact station dataset, but larger deployments can benefit from GPU indexing and batch embedding.

Recommended production approach:

1. Build embeddings in controlled offline jobs.
2. Persist the generated FAISS index and metadata table.
3. Load the prebuilt index at application startup.
4. Reserve GPU usage for large rebuilds, not for every user request unless query volume requires it.

## Key Results and Robustness

The notebook includes a retrieval evaluation suite with representative English and Arabic queries covering:

- Direct fact retrieval: station type and line assignment.
- Relational retrieval: previous and next station questions.
- Aggregation stress testing: station counts by line and station type.
- Bilingual query behavior: Arabic and English user input.

No fabricated benchmark metrics are reported in this repository. Final precision, recall, latency, and answer-accuracy metrics should be generated after running the notebook against the approved production dataset and the enabled Bedrock model. This is the correct evaluation approach because retrieval metrics depend on the exact dataset version, station naming conventions, and approved ground-truth answers.

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
- Missing dataset files raise explicit configuration errors.
- Missing required columns raise schema validation errors.
- AWS or Bedrock failures return a controlled service-availability message in the UI.
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
- Add cross-encoder reranking for higher precision on ambiguous station names.
- Add hybrid search combining keyword filters and vector retrieval.
- Implement deterministic handlers for analytics questions before LLM generation.
- Add an API layer with authentication and rate limiting.
- Add CI checks for notebook execution, linting, and unit tests.

## License

Add the appropriate license for the dataset and project code before publishing the repository.
