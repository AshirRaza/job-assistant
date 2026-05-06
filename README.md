# CV Analyzer (RAG Pipeline)

Backend service that compares a CV against a job description using a multi-stage pipeline:

- ingestion + chunking
- NER skill extraction
- vector retrieval (Chroma)
- optional reranking
- LLM analysis (Claude) or offline heuristic mode

## Project Structure

- `main.py` - FastAPI app and endpoints
- `pipeline/` - ingestion, embeddings, vector store, reranker, NER, LLM, RAG orchestration
- `models/schemas.py` - Pydantic contracts
- `utils/` - settings and helpers
- `evaluation/` - evaluation script + sample dataset
- `tests/` - API and NER tests

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m spacy download en_core_web_sm
```

Copy and edit environment config:

```bash
cp .env.example .env
```

## Environment Variables

- `ANTHROPIC_API_KEY` - required for online Claude mode
- `CLAUDE_MODEL` - Claude model name
- `CHROMA_PERSIST_DIR` - Chroma persistence directory
- `EMBEDDING_MODEL` - sentence-transformers embedding model
- `RERANKER_MODEL` - cross-encoder model
- `SPACY_MODEL` - spaCy model
- `USE_RERANKER` - enable/disable reranker
- `OFFLINE_MODE` - `true` to avoid external model/API calls
- `MAX_CONTEXT_TOKENS` - RAG context token budget
- `SUMMARIZE_OVERFLOW` - summarize overflow context chunks
- `OVERFLOW_SUMMARY_TOKENS` - token cap for overflow summary

## Run API

```bash
venv/bin/uvicorn main:app --reload
```

## Endpoints

- `GET /health` - liveness check
- `GET /ready` - readiness diagnostics (pipeline init + key presence)
- `POST /analyze` - full analysis response
- `POST /score` - match score only

### Analyze Example (text)

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F 'cv_text=Python FastAPI Docker AWS engineer with SQL experience' \
  -F 'jd_text=Need Python FastAPI AWS engineer with Kubernetes and SQL' \
  -F 'top_k=5'
```

### Analyze Example (files)

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "cv_file=@/path/to/cv.pdf" \
  -F "jd_file=@/path/to/jd.pdf" \
  -F 'top_k=5'
```

## Offline vs Online Mode

### Offline mode (`OFFLINE_MODE=true`)

- uses deterministic local embeddings
- disables cross-encoder reranker model loading
- uses heuristic analyzer instead of Anthropic API
- works on restricted networks

### Online mode (`OFFLINE_MODE=false`)

- uses HuggingFace models for embeddings/reranking
- uses Claude API for final analysis
- requires outbound network access to HuggingFace and Anthropic

## Evaluation

Run the simple evaluation script:

```bash
venv/bin/python evaluation/ragas_eval.py
```

Uses `evaluation/sample_dataset.json` and reports:

- average score absolute error
- missing-skill recall
- matched-skill precision

## Tests

```bash
venv/bin/pytest -q
```

## Troubleshooting

- **`403 Forbidden` during model/API init**
  - proxy/network is blocking HuggingFace and/or Anthropic
  - use `OFFLINE_MODE=true` or fix network allowlist
- **`Temporary failure in name resolution`**
  - DNS/network routing issue in current environment
- **Invalid PDF ingestion errors**
  - ensure uploaded files are valid text PDFs
