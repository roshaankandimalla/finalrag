# FinalRAG: Multimodal Hybrid RAG Portfolio Project

FinalRAG is a local, domain-aware multimodal RAG system for finance, legal, and medical documents. It supports PDF, HTML, and CSV sources, converts them into a common document schema, builds hierarchical parent-child chunks, indexes them with dense multimodal embeddings and SPLADE sparse vectors, retrieves with hybrid search, reranks with Voyage, generates grounded answers with Gemini, and evaluates quality with RAGAS.

The goal is to demonstrate an enterprise-style RAG pipeline, not a simple vector-search demo.

## What This Project Shows

- Domain-aware ingestion for finance, legal, and medical files
- PDF parsing with LlamaParse
- HTML parsing with Firecrawl / Trafilatura
- Large CSV processing with Pandas
- Normalization into a shared element schema
- Hierarchical parent-child chunking
- Multimodal retrieval chunks with text, tables, and image paths
- Voyage multimodal dense embeddings
- SPLADE sparse vectors for lexical retrieval
- PostgreSQL + pgvector storage through Docker
- Hybrid retrieval with Reciprocal Rank Fusion
- Child-to-parent context expansion
- Voyage reranking
- Gemini grounded answer generation
- Citations with file, page, section, and chunk metadata
- RAGAS evaluation with saved metrics

## Architecture

```text
Local domain files
  -> file discovery
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

## End-To-End Pipeline

```text
LOCAL DOMAIN FILES
data/input/
  legal/
    *.pdf / *.html / *.csv
  finance/
    *.pdf / *.html / *.csv
  medical/
    *.pdf / *.html / *.csv
      |
      v
DOMAIN-AWARE FILE DISCOVERY
      |
      |-- Read domain folder name
      |-- Assign domain
      |-- Assign document_id
      |-- Detect source_type
      |-- Save original file path
      |
      v
DOCUMENT REGISTRY
PostgreSQL: documents table
      |
      |-- document_id
      |-- domain
      |-- file_name
      |-- source_type
      |-- file_path
      |-- parser_used
      |-- status
      |-- created_at
      |
      v
FILE ROUTER
      |
      |-- PDF  ---> LlamaParse
      |-- HTML ---> Firecrawl / Trafilatura
      |-- CSV  ---> Pandas
      |
      v
RAW PARSING
      |
      |-- PDF:
      |     |-- Markdown text
      |     |-- Tables
      |     |-- Images
      |     |-- Page metadata
      |     |-- OCR text if LlamaParse provides it
      |
      |-- HTML:
      |     |-- Markdown
      |     |-- HTML
      |     |-- Structured sections
      |     |-- Metadata
      |
      |-- CSV:
            |-- Raw rows
            |-- Hospital profiles
            |-- Hospital category documents
            |-- Retrieval text
      |
      v
SAVE RAW PARSED OUTPUTS
Local disk
      |
      |-- data/parsed/{domain}/pdf/
      |-- data/parsed/{domain}/html/
      |-- data/parsed/{domain}/csv/
      |
      v
NORMALIZATION LAYER
Convert parser-specific outputs into common elements
      |
      |-- Text element
      |-- Table element
      |-- Image element
      |-- Metadata element
      |
      v
NORMALIZED ELEMENT SCHEMA
      |
      |-- element_id
      |-- document_id
      |-- domain
      |-- source_type
      |-- file_name
      |-- element_type: text / table / image
      |-- text
      |-- table_markdown
      |-- table_html
      |-- image_path
      |-- page_number
      |-- row_range
      |-- section_title
      |-- metadata
      |
      v
IMAGE HANDLING
      |
      |-- Decode parser image payloads when available
      |-- Save images locally
      |-- Store image_path in chunks and database metadata
      |-- Create image_base64 only temporarily for Gemini generation
      |
      v
HIERARCHICAL MULTIMODAL CHUNKING
      |
      |-- Build document hierarchy
      |-- Detect headings and sections
      |-- Create parent sections
      |-- Create child retrieval chunks
      |-- Preserve domain
      |-- Preserve page numbers / row ranges
      |-- Preserve table and image relationships
      |
      v
DOCUMENT HIERARCHY
      |
      Document
        |
        +-- Domain: legal / finance / medical
              |
              +-- Parent Section
                    |
                    +-- Child Retrieval Chunk
                    +-- Child Retrieval Chunk
                    +-- Child Retrieval Chunk
      |
      v
PARENT SECTIONS
PostgreSQL: parent_sections table
      |
      |-- parent_id
      |-- document_id
      |-- domain
      |-- source_type
      |-- section_title
      |-- section_path
      |-- page_numbers
      |-- row_ranges
      |-- parent_text
      |-- metadata
      |
      v
CHILD RETRIEVAL CHUNKS
Main retrieval unit
      |
      |-- chunk_id
      |-- parent_id
      |-- document_id
      |-- domain
      |-- source_type
      |-- file_name
      |-- page_numbers
      |-- row_range
      |-- section_title
      |-- modalities
      |-- text
      |-- table_markdown
      |-- table_html
      |-- image_paths
      |-- metadata
      |
      v
INDEX CHILD CHUNKS ONLY
      |
      +------------------------------------------------+
      |                                                |
      v                                                v
DENSE EMBEDDING INPUT                         SPARSE EMBEDDING INPUT
Voyage multimodal                             SPLADE text sparse model
      |                                                |
      |-- domain                                       |-- domain
      |-- section_title                                |-- section_title
      |-- text                                         |-- text
      |-- table_markdown                               |-- table_markdown
      |-- PIL image from image_path                    |-- source metadata text
      |                                                |
      v                                                v
VOYAGE MULTIMODAL EMBEDDING                  SPLADE SPARSE EMBEDDING
      |                                                |
      |-- dense_embedding vector                       |-- sparse_embedding sparsevec
      |                                                |
      +------------------------+-----------------------+
                               |
                               v
DOCKER POSTGRESQL + PGVECTOR
      |
      |-- documents
      |-- parent_sections
      |-- child_chunks
      |-- hospital_profiles
      |-- hospital_category_docs
      |-- dense_embedding vector(...)
      |-- sparse_embedding sparsevec(...)
      |-- domain metadata
      |-- metadata JSONB
      |
      v
INDEXING COMPLETE
Document status = completed
```

## Why Multimodal

The system keeps different evidence types together instead of flattening everything into plain text.

- Text is used directly for retrieval and generation.
- Tables are represented as markdown for retrieval and HTML for generation.
- Images are stored as local file paths and loaded only when needed.
- Image base64 is created temporarily during generation and is not stored in PostgreSQL.

This keeps storage efficient while still allowing multimodal reasoning.

## Project Layout

```text
src/finalrag/
  config.py
  models.py
  discovery/       file discovery
  parsing/         PDF, HTML, CSV parsing
  normalization/   parser-output normalization
  chunking/        hierarchical chunking
  embeddings/      Voyage and SPLADE embedding code
  database/        PostgreSQL connection, schema, repository layer
  retrieval/       dense, sparse, RRF, parent expansion, reranking
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

`parent_sections` stores larger context sections for expansion after retrieval.

`child_chunks` stores the indexed retrieval units, dense vectors, sparse vectors, table metadata, image paths, and citation metadata.

For the medical CSV, `hospital_profiles` and `hospital_category_docs` avoid reconstructing hospital-level evidence from raw measure rows during every query.

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

The main pipeline uses the original user query. Query rewriting is intentionally not required, which keeps retrieval behavior easier to inspect and evaluate.

## Setup

Create a local environment and install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in local credentials:

```powershell
copy .env.example .env
```

Required services and keys:

```text
DATABASE_URL
LLAMA_CLOUD_API_KEY or LLAMA_CLOUD_API_KEY_1...
FIRECRAWL_API_KEY
VOYAGE_API_KEY
GEMINI_API_KEY or GEMINI_API_KEY_1...
```

Do not commit `.env`.

## Local PostgreSQL + pgvector

Start PostgreSQL with Docker:

```powershell
docker compose up -d db
```

Create the schema:

```powershell
python scripts/01_create_database.py
```

Useful commands:

```powershell
docker compose ps
docker compose logs -f db
docker compose stop db
docker compose start db
```

Do not run `docker compose down -v` unless you intentionally want to delete the PostgreSQL volume.

## Build Pipeline

Run the pipeline in order:

```powershell
python scripts/02_discover_files.py
python scripts/03_parse_all.py
python scripts/04_normalize_elements.py
python scripts/05_create_chunks.py
python scripts/06_index_chunks.py
```

What each step does:

```text
02_discover_files.py      registers source files in PostgreSQL
03_parse_all.py           parses PDF, HTML, and CSV sources
04_normalize_elements.py  converts parser outputs into common elements
05_create_chunks.py       creates parent sections and child retrieval chunks
06_index_chunks.py        stores chunks, dense embeddings, SPLADE vectors, and HNSW indexes
```

If SPLADE vectors were generated externally, import them because splade required gpu so run them in google colab and add them  into project folder and send it to the database 
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

Fast mode uses smaller retrieval limits but still keeps Voyage reranking:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --fast
```

Inspect retrieval without Gemini:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --retrieval-only
```

Print complete retrieved parent and child contexts:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --retrieval-only --full
```

## Evaluation

RAGAS metrics:

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

This version is local-first.

No frontend.

No FastAPI backend.

No authentication.

No cloud deployment.

The interface is notebook or terminal based.

## Why This Matters

FinalRAG demonstrates the engineering details that matter in practical RAG systems:

- handling mixed document formats
- preserving source metadata
- supporting text, tables, and images
- combining semantic and lexical retrieval
- using parent-child context expansion
- grounding answers with citations
- evaluating quality with RAGAS
- keeping the pipeline reproducible with scripts and tests
