# FinalRAG: Multimodal Hybrid RAG System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Local%20Stack-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Multimodal-7C3AED?style=for-the-badge)
![Evaluation](https://img.shields.io/badge/Evaluation-RAGAS-00A67E?style=for-the-badge)

FinalRAG is a local-first, domain-aware **multimodal hybrid RAG system** for finance, legal, and medical documents. It ingests PDF, HTML, and CSV sources, normalizes them into a shared schema, builds hierarchical parent-child chunks, indexes dense and sparse representations, retrieves with hybrid search, reranks evidence, generates grounded answers, and evaluates quality with RAGAS.

This project is designed to demonstrate an enterprise-style RAG pipeline rather than a simple vector search demo.

## Highlights

- Domain-aware ingestion for finance, legal, and medical data
- PDF parsing with LlamaParse
- HTML parsing with Firecrawl and Trafilatura
- Large CSV processing with Pandas
- Shared normalized schema for text, tables, images, and metadata
- Hierarchical parent-child chunking for better context expansion
- Multimodal dense embeddings with Voyage
- SPLADE sparse vectors for lexical retrieval
- PostgreSQL + pgvector storage through Docker
- Hybrid retrieval with Reciprocal Rank Fusion
- Child-to-parent context expansion
- Voyage reranking
- Gemini grounded answer generation
- Citations with file, page, section, and chunk metadata
- RAGAS evaluation with saved metrics
- Unit tests for core pipeline components

## Quick Start

```powershell
pip install -r requirements.txt
copy .env.example .env
docker compose up -d db
python scripts/01_create_database.py
python scripts/02_discover_files.py
python scripts/03_parse_all.py
python scripts/04_normalize_elements.py
python scripts/05_create_chunks.py
python scripts/06_index_chunks.py
python scripts/07_query.py "What was Reliance Industries financial performance?"
```

Before running the full pipeline, fill in the required API keys and database URL in `.env`.

## Architecture

```text
Local domain files
  -> domain-aware file discovery
  -> parser router
      -> PDF: LlamaParse
      -> HTML: Firecrawl / Trafilatura
      -> CSV: Pandas
  -> raw parsed outputs
  -> normalized elements
  -> hierarchical parent-child chunks
  -> dense embeddings with Voyage multimodal
  -> sparse embeddings with SPLADE
  -> PostgreSQL + pgvector
  -> hybrid retrieval
      -> dense retrieval
      -> sparse retrieval
      -> Reciprocal Rank Fusion
  -> child-to-parent expansion
  -> Voyage reranker
  -> Gemini answer generation
  -> grounded answer with citations
  -> RAGAS evaluation
```

## Pipeline

FinalRAG follows a reproducible script-based workflow.

```text
data/input/
  legal/      *.pdf / *.html / *.csv
  finance/    *.pdf / *.html / *.csv
  medical/    *.pdf / *.html / *.csv
      |
      v
File discovery and document registry
      |
      v
Parser routing by source type
      |
      v
Raw parsing outputs
      |
      v
Normalized elements
      |
      v
Parent sections + child retrieval chunks
      |
      v
Dense and sparse embedding generation
      |
      v
PostgreSQL + pgvector indexing
      |
      v
Hybrid retrieval + reranking
      |
      v
Grounded answer generation + citations
      |
      v
RAGAS evaluation
```

## Why Multimodal

FinalRAG keeps different evidence types connected instead of flattening every source into plain text.

- Text is used directly for retrieval and generation.
- Tables are represented as markdown for retrieval and HTML for answer generation.
- Images are stored as local file paths and loaded only when needed.
- Image base64 payloads are created temporarily during generation and are not stored in PostgreSQL.

This keeps storage efficient while preserving multimodal reasoning capability.

## Project Layout

```text
src/finalrag/
  config.py
  models.py
  discovery/       file discovery
  parsing/         PDF, HTML, and CSV parsing
  normalization/   parser-output normalization
  chunking/        hierarchical chunking
  embeddings/      Voyage and SPLADE embedding code
  database/        PostgreSQL connection, schema, and repository layer
  retrieval/       dense search, sparse search, RRF, parent expansion, reranking
  generation/      context building, citations, Gemini generation
  evaluation/      RAGAS evaluation
  graphing/        unified graph tracing utilities

scripts/
  01_create_database.py
  02_discover_files.py
  03_parse_all.py
  04_normalize_elements.py
  05_create_chunks.py
  06_index_chunks.py
  07_query.py
  08_evaluate.py
  09_build_unified_graph.py
  10_export_embeddings.py
  colab_generate_splade.py
  test_llamaparse_one_pdf.py

tests/
  unit tests for discovery, parsing, normalization, chunking,
  retrieval, database helpers, citations, and graphing
```

## Data Layout

```text
data/
  input/        original local source files
  parsed/       raw parser outputs
  normalized/   common element JSONL files
  chunks/       parent and child chunk JSONL files
  images/       extracted images
  embeddings/   exported dense and sparse embedding backups
  eval/         evaluation questions and selected result files
```

## Database Design

Core tables:

```text
documents
parent_sections
child_chunks
hospital_profiles
hospital_category_docs
queries
citations
```

`documents` stores one row per source file.

`parent_sections` stores larger context sections used for expansion after retrieval.

`child_chunks` stores the indexed retrieval units, dense vectors, sparse vectors, table metadata, image paths, and citation metadata.

For the medical CSV workflow, `hospital_profiles` and `hospital_category_docs` prevent the system from rebuilding hospital-level evidence from raw measure rows during every query.

## Retrieval Flow

```text
User query
  -> domain centroid routing
  -> Voyage dense retrieval
  -> SPLADE sparse retrieval
  -> Reciprocal Rank Fusion
  -> child-to-parent expansion
  -> parent deduplication
  -> Voyage reranking
  -> Gemini context builder
  -> grounded answer with citations
```

The main pipeline uses the original user query. Query rewriting is intentionally optional, keeping retrieval behavior easier to inspect and evaluate.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- PostgreSQL with pgvector, provided through the included Docker setup
- API keys for the configured parsers, embedding providers, reranker, and generation model

Required environment variables:

```text
DATABASE_URL
LLAMA_CLOUD_API_KEY or LLAMA_CLOUD_API_KEY_1...
FIRECRAWL_API_KEY
VOYAGE_API_KEY
GEMINI_API_KEY or GEMINI_API_KEY_1...
```

Do not commit `.env`.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your local environment file:

```powershell
copy .env.example .env
```

Start PostgreSQL with Docker:

```powershell
docker compose up -d db
```

Create the database schema:

```powershell
python scripts/01_create_database.py
```

Useful Docker commands:

```powershell
docker compose ps
docker compose logs -f db
docker compose stop db
docker compose start db
```

Avoid `docker compose down -v` unless you intentionally want to delete the PostgreSQL volume.

## Build the Index

Run the pipeline in order:

```powershell
python scripts/02_discover_files.py
python scripts/03_parse_all.py
python scripts/04_normalize_elements.py
python scripts/05_create_chunks.py
python scripts/06_index_chunks.py
```

What each step does:

| Script | Purpose |
|---|---|
| `02_discover_files.py` | Registers source files in PostgreSQL |
| `03_parse_all.py` | Parses PDF, HTML, and CSV sources |
| `04_normalize_elements.py` | Converts parser outputs into common elements |
| `05_create_chunks.py` | Creates parent sections and child retrieval chunks |
| `06_index_chunks.py` | Stores chunks, dense embeddings, SPLADE vectors, and HNSW indexes |

## SPLADE Workflow

SPLADE sparse vector generation can be GPU-heavy. If vectors are generated externally, such as in Google Colab, export them as JSONL and import them into the database:

```powershell
python scripts/06_index_chunks.py --import-splade-jsonl path --splade
```

Rebuild domain centroids after changing dense embeddings:

```powershell
python scripts/06_index_chunks.py --skip-store --domain-centroids
```

## Querying

Run the full retrieval and generation pipeline:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?"
```

Use fast mode for smaller retrieval limits while still keeping Voyage reranking:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --fast
```

Inspect retrieval without Gemini generation:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --retrieval-only
```

Print complete retrieved parent and child contexts:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --retrieval-only --full
```

## Evaluation

RAGAS metrics used in this project:

```text
answer_relevancy
context_precision
context_recall
faithfulness
```

Run a smoke test:

```powershell
python scripts/08_evaluate.py --limit 1 --top-k 5 --output data/eval/smoke_results.json
```

Run a full evaluation:

```powershell
python scripts/08_evaluate.py --questions data/eval/questions.json --output data/eval/results.json --top-k 8 --batch-size 1 --batch-sleep 60 --max-retries 10 --timeout 400 --resume
```

Run the medical CSV evaluation set:

```powershell
python scripts/08_evaluate.py --questions data/eval/medical_csv_questions_15.json --output data/eval/medical_csv_results_15.json --top-k 8 --batch-size 1 --batch-sleep 60 --max-retries 10 --timeout 400 --resume
```

The evaluator can use multiple Gemini keys. Each key processes one question at a time, sleeps between requests, retries transient failures, checkpoints progress, and writes per-question metrics plus aggregate summaries.

## Tests

Run all tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Current Scope

FinalRAG is currently a local-first research and engineering project with a CLI/notebook interface. The current version focuses on the ingestion, indexing, retrieval, generation, citation, and evaluation pipeline.

Planned production-facing layers such as a frontend, FastAPI backend, authentication, and cloud deployment are outside the current scope.

## Why This Matters

FinalRAG demonstrates the engineering details that matter in practical RAG systems:

- Handling mixed document formats
- Preserving source metadata
- Supporting text, tables, and images
- Combining semantic and lexical retrieval
- Using parent-child context expansion
- Grounding answers with citations
- Evaluating quality with RAGAS
- Keeping the pipeline reproducible with scripts and tests

## Author

**Roshaankandimalla**

GitHub: [roshaankandimalla](https://github.com/roshaankandimalla)
