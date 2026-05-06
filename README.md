# CV Analyzer (RAG Pipeline)

## Purpose

This project is a backend service that helps measure how well a candidate CV matches a job
description and explains the gaps in a structured, machine-readable way. It is designed for
resume screening, candidate feedback tooling, and recruiter-assist workflows where you need
consistent scoring plus actionable recommendations.

## What It Does

Given a CV and a JD (as raw text or PDF), the service:

- parses and chunks documents
- extracts and normalizes skills
- retrieves the most relevant CV evidence for JD requirements
- optionally reranks evidence for better precision
- generates a structured analysis response with:
  - match score
  - reasoning
  - matched/missing/transferable skills
  - CV rewrite suggestions
  - keyword gaps and final verdict

Core processing stages:

- ingestion + chunking
- NER skill extraction
- vector retrieval (Chroma)
- optional reranking
- LLM analysis (Claude) or offline heuristic mode

## Architecture and Workflow

This service implements a retrieval-augmented analysis pipeline for CV vs JD matching.
The API layer is intentionally thin: it validates input, resolves text from files, and
delegates all orchestration to `pipeline/rag.py`.

### 1) Request ingestion

- `POST /analyze` and `POST /score` accept either:
  - raw form text (`cv_text`, `jd_text`)
  - uploaded PDFs (`cv_file`, `jd_file`)
- Uploaded files are parsed with `pdfplumber` and converted into page-like records.
- Validation is enforced by `AnalysisRequest` in `models/schemas.py`:
  each side must provide text or file input.

### 2) Chunking and metadata

- `pipeline/ingestion.py` chunks each document using
  `RecursiveCharacterTextSplitter` (`chunk_size=512`, `chunk_overlap=50`).
- Every chunk contains metadata used downstream:
  - `source` (`cv` / `jd`)
  - `page_number`
  - `chunk_index`
  - `section_type` (`skills`, `experience`, `projects`, `education`, `summary`, `general`)
- Section hints improve retrieval relevance and context readability.

### 3) Skill extraction (NER layer)

- `pipeline/ner.py` extracts and normalizes skill candidates from full CV/JD text.
- Alias normalization (example: `js -> javascript`) and noise filtering reduce
  low-signal terms before scoring and explanation.

### 4) Embeddings and vector indexing

- `pipeline/embeddings.py` computes embeddings and caches vectors by text hash.
- In online mode, it loads `EMBEDDING_MODEL`.
- In offline mode, it falls back to deterministic local embeddings so the pipeline
  remains runnable without external downloads.
- `pipeline/vector_store.py` persists chunk vectors in Chroma (`CHROMA_PERSIST_DIR`).
- Each CV run gets a `doc_id`; retrieval filters by this id to avoid cross-document leakage.

### 5) Retrieval and ranking

- JD chunks query the CV index with cosine-space nearest neighbors.
- Retrieval uses a hybrid score:
  - dense vector relevance
  - lexical token overlap with the JD query
  - section-based weighting (experience/projects/skills prioritized)
- Candidate pool is intentionally larger than final output:
  retrieve-more first, then cut to final top-k.
- If enabled, `pipeline/reranker.py` applies a cross-encoder reranker for final ordering.

### 6) Context packing and analysis

- `pipeline/rag.py` builds an LLM context window from top chunks with token budgeting.
- Overflow chunks can be summarized if `SUMMARIZE_OVERFLOW=true`.
- Final analysis happens in `pipeline/llm.py`:
  - online mode: calls Claude with strict JSON prompt contract
  - offline mode: heuristic fallback analysis (schema-compatible)
- Output is validated against `AnalysisResponse` before returning.

### 7) Error handling and observability

- API errors use a consistent `ErrorResponse` envelope.
- Common failure classes:
  - validation errors (`422`)
  - ingestion errors (`400`)
  - pipeline/LLM errors (`400` / `500`)
- Request middleware adds per-request timing and request IDs.
- Stage-level timings are logged from the RAG pipeline.

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

### Request contracts

- `top_k` controls final number of evidence chunks passed forward.
- `/score` internally uses a lightweight path (`top_k=1`) and returns only:
  - `{"match_score": <int 0-100>}`

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

You can also evaluate against larger datasets by passing a different path to
`run_eval(...)` from `evaluation/ragas_eval.py`.

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
- **Unexpected low-quality reasoning**
  - check whether `OFFLINE_MODE=true`; this uses heuristic output instead of Claude
