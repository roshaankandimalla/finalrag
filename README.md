# Multimodal Hybrid RAG System

A local, domain-aware multimodal RAG project over PDF, HTML, and CSV files.

This project is designed as a recruiter-facing portfolio project that demonstrates:

- multimodal document parsing
- hierarchical parent-child chunking
- dense multimodal retrieval
- SPLADE sparse retrieval
- hybrid search with Reciprocal Rank Fusion
- Voyage reranking
- grounded Gemini answer generation
- citation tracking
- RAGAS evaluation

## Overview

The system works over local files from multiple domains:

```text
legal
finance
medical
```

Supported file types:

```text
PDF
HTML
CSV
```

The pipeline parses every file, normalizes all parser outputs into a common schema, builds hierarchical multimodal chunks, indexes child chunks using both dense and sparse vectors, retrieves with hybrid search, reranks candidates, and generates grounded answers with citations.

## Tech Stack

| Layer | Tool |
|---|---|
| PDF parsing | LlamaParse |
| HTML parsing | Trafilatura / Firecrawl |
| CSV parsing | Pandas |
| Chunking | Hierarchical parent-child chunking |
| Dense embeddings | Voyage multimodal embeddings |
| Sparse retrieval | SPLADE |
| Vector database | Docker PostgreSQL + pgvector |
| Reranking | Voyage reranker |
| Generation | Gemini |
| Evaluation | RAGAS |
| Interface | Notebook / terminal |

## High-Level Architecture

```text
LOCAL DOMAIN FILES
PDF / HTML / CSV
      |
      v
DOMAIN-AWARE FILE DISCOVERY
      |
      v
FILE ROUTER
      |
      |-- PDF  -> LlamaParse
      |-- HTML -> Trafilatura / Firecrawl
      |-- CSV  -> Pandas
      |
      v
RAW PARSED OUTPUTS
      |
      v
NORMALIZED ELEMENTS
      |
      v
HIERARCHICAL PARENT-CHILD CHUNKING
      |
      v
INDEX CHILD CHUNKS ONLY
      |
      |-- Voyage multimodal dense embeddings
      |-- SPLADE sparse vectors
      |
      v
DOCKER POSTGRESQL + PGVECTOR
      |
      v
HYBRID RETRIEVAL
      |
      |-- Dense retrieval
      |-- Sparse retrieval
      |-- Reciprocal Rank Fusion
      |
      v
CHILD -> PARENT EXPANSION
      |
      v
VOYAGE RERANKER
      |
      v
GEMINI CONTEXT BUILDER
      |
      v
GEMINI GENERATION
      |
      v
GROUNDED ANSWER + CITATIONS
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
      |-- HTML ---> Trafilatura / Firecrawl
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
      |     |-- Clean main text
      |     |-- Title
      |     |-- Metadata
      |
      |-- CSV:
            |-- DataFrame
            |-- Columns
            |-- Row groups
            |-- Table markdown
            |-- Table HTML
      |
      v
SAVE RAW PARSED OUTPUTS
Local disk
      |
      |-- data/parsed/{domain}/pdf/{document_id}.md
      |-- data/parsed/{domain}/pdf/{document_id}.json
      |-- data/parsed/{domain}/html/{document_id}.json
      |-- data/parsed/{domain}/csv/{document_id}.json
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
      |-- If parser returns image bytes/base64:
      |       decode image
      |       save image file
      |       store image_path only
      |
      |-- Store images:
      |       data/images/{domain}/{document_id}/image_001.png
      |
      |-- Do not store image_base64 in Postgres
      |
      v
HIERARCHICAL MULTIMODAL CHUNKING
      |
      |-- Build document hierarchy
      |-- Detect headings/sections
      |-- Create parent sections
      |-- Create child retrieval chunks
      |-- Preserve domain
      |-- Preserve page numbers / row ranges
      |-- Preserve table/image relationships
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
      |-- PIL image from image_path                    |-- file/source metadata text
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
      |-- dense_embedding vector(...)
      |-- sparse_embedding sparsevec(...)
      |-- domain metadata
      |-- metadata JSONB
      |
      v
INDEXING COMPLETE
Document status = completed
```

## Query-Time Pipeline

```text
USER QUERY
      |
      v
ORIGINAL QUERY EMBEDDING
      |
      v
DOMAIN CENTROID ROUTING
      |
      |-- High confidence: search one domain
      |-- Ambiguous: search top two domains
      |-- Low confidence: search all domains
      |
      +------------------------------------------------+
      |                                                |
      v                                                v
DENSE QUERY EMBEDDING                         SPARSE QUERY EMBEDDING
Voyage multimodal                             SPLADE
      |                                                |
      v                                                v
DENSE RETRIEVAL                               SPLADE RETRIEVAL
PostgreSQL pgvector                                 PostgreSQL sparsevec
      |                                                |
      |-- top 50 child chunks                          |-- top 50 child chunks
      |-- apply domain filter                          |-- apply domain filter
      |-- apply source_type filter                     |-- apply source_type filter
      |-- dense score                                  |-- sparse score
      |                                                |
      +------------------------+-----------------------+
                               |
                               v
RECIPROCAL RANK FUSION
      |
      |-- Merge dense results + sparse results
      |-- Deduplicate by chunk_id
      |-- RRF score = 1/(60 + dense_rank) + 1/(60 + sparse_rank)
      |-- Select top fused child chunks
      |
      v
TOP FUSED CHILD CHUNKS
      |
      v
CHILD -> PARENT EXPANSION
      |
      |-- Fetch parent_section for each child chunk
      |-- Keep matched child as primary evidence
      |-- Attach parent section as supporting context
      |-- Preserve domain/source metadata
      |
      v
DEDUPLICATE PARENTS
      |
      |-- Avoid repeating same parent section too many times
      |-- Keep strongest matched children per parent
      |
      v
PARENT + MATCHED CHILD CONTEXTS
      |
      v
VOYAGE RERANKER
      |
      |-- Input:
      |     user query
      |
      |-- Candidates:
      |     domain
      |     source_type
      |     parent section title
      |     child text
      |     table_markdown
      |     page metadata / row range
      |
      |-- Output:
      |     reranked top contexts
      |
      v
TOP CONTEXTS
      |
      v
GEMINI CONTEXT BUILDER
      |
      |-- For each selected context:
      |     source_id
      |     domain
      |     file_name
      |     source_type
      |     page_numbers
      |     row_range
      |     section_title
      |     child text
      |     parent context
      |     table_html
      |     image_base64 from image_path
      |
      v
CITATION ASSEMBLY
      |
      |-- source_id
      |-- domain
      |-- document_id
      |-- file_name
      |-- source_type
      |-- page_number / row range / section
      |-- parent_id
      |-- chunk_id
      |-- dense_score
      |-- sparse_score
      |-- rrf_score
      |-- rerank_score
      |
      v
GEMINI GENERATION
      |
      |-- Grounded prompt
      |-- User query
      |-- Top contexts
      |-- Text evidence
      |-- Table HTML
      |-- Image base64
      |-- Citation instructions
      |
      v
GROUNDED ANSWER
      |
      |-- answer
      |-- citations [1], [2], [3]
      |-- domain-aware source references
      |-- insufficient evidence message if needed
      |
      v
SAVE QUERY TRACE
Docker PostgreSQL
      |
      |-- query
      |-- selected domain filter
      |-- answer
      |-- retrieved chunks
      |-- fused chunks
      |-- reranked chunks
      |-- citations
      |-- models used
      |-- latency
      |
      v
NOTEBOOK / TERMINAL OUTPUT
      |
      |-- Final answer
      |-- Citation list
      |-- Domain/source labels
      |-- Retrieved source snippets
      |-- Table previews
      |-- Image paths
      |-- Scores/debug info
```

## Data Layout

```text
data/
  input/
    legal/
    finance/
    medical/

  parsed/
    legal/
      pdf/
      html/
      csv/
    finance/
      pdf/
      html/
      csv/
    medical/
      pdf/
      html/
      csv/

  images/
    legal/
    finance/
    medical/

  eval/
    questions.json
    results.json
```

## Database Tables

### documents

Stores one row per source file.

```text
document_id
domain
file_name
source_type
file_path
parser_used
parser_version
status
created_at
```

### parent_sections

Stores larger context sections.

```text
parent_id
document_id
domain
source_type
section_title
section_path
page_numbers
row_ranges
parent_text
metadata
```

### child_chunks

Stores indexed retrieval chunks.

```text
chunk_id
parent_id
document_id
domain
source_type
file_name
page_numbers
row_range
section_title
modalities
text
table_markdown
table_html
image_paths
metadata
dense_embedding
sparse_embedding
created_at
```

### queries

Stores user query traces.

```text
query_id
question
domain_filter
source_type_filter
answer
generator_model
latency
created_at
```

### citations

Stores answer citations and retrieval scores.

```text
citation_id
query_id
document_id
parent_id
chunk_id
source_id
domain
file_name
source_type
page_numbers
row_range
section_title
dense_score
sparse_score
rrf_score
rerank_score
```

## Chunking Strategy

The system uses hierarchical parent-child chunking.

```text
Parent section = larger section context
Child chunk = smaller retrieval evidence
```

Only child chunks are indexed.

Parent sections are used after retrieval for context expansion.

Supported child chunk types:

```text
text-only
table-only
image-only
text + table
text + image
text + table + image
```

Chunking rules:

- preserve domain metadata
- preserve source type
- preserve page numbers or CSV row ranges
- preserve section titles
- keep tables intact when possible
- keep related text/table/image content together
- avoid mixing unrelated page elements
- store image paths, not base64

## Retrieval Strategy

The system uses hybrid retrieval.

### Dense Retrieval

Dense retrieval uses Voyage multimodal embeddings.

Embedding input can include:

```text
text
table markdown
PIL image loaded from image_path
```

Dense retrieval is responsible for semantic and multimodal matching.

### Sparse Retrieval

Sparse retrieval uses SPLADE.

SPLADE input is text-only:

```text
domain
section title
source metadata
text
table markdown
```

Sparse retrieval helps with:

- exact terms
- names
- numbers
- domain-specific vocabulary
- CSV values
- legal/medical/finance terminology

### Fusion

Dense and sparse results are fused using Reciprocal Rank Fusion.

```text
rrf_score = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)
```

### Reranking

After fusion, Voyage reranker selects the final contexts for Gemini.

Reranker candidates include:

```text
domain
source type
file name
section title
page numbers or row range
child text
table markdown
parent context
```

## Generation Strategy

Gemini receives the top contexts after reranking.

Each context includes:

```text
source_id
domain
file_name
source_type
page_numbers
row_range
section_title
child text
parent context
table_html
image_base64 generated from image_path
```

Gemini is instructed to:

- answer only from provided evidence
- cite sources using `[1]`, `[2]`, etc.
- say when evidence is insufficient
- avoid inventing facts
- use text, tables, and images together

## Evaluation

The project uses RAGAS for evaluation.

Metrics:

```text
faithfulness
answer relevancy
context precision
context recall
```

Evaluation questions are domain-aware:

```text
legal questions
finance questions
medical questions
cross-domain questions
```

Questions live in `data/eval/questions.json`. Each item requires a question and
a human-verified reference answer:

```json
{
  "id": "finance-ril-performance",
  "domain": "finance",
  "question": "What was Reliance Industries financial performance in fiscal year 2024-25?",
  "reference_answer": "Reliance Industries reported ..."
}
```

Run a one-question smoke evaluation:

```powershell
python scripts/08_evaluate.py --limit 1 --top-k 5 --output data/eval/smoke_results.json
```

Configure up to six Gemini keys for parallel evaluation:

```dotenv
GEMINI_API_KEY_1=
GEMINI_API_KEY_2=
GEMINI_API_KEY_3=
GEMINI_API_KEY_4=
GEMINI_API_KEY_5=
GEMINI_API_KEY_6=
```

Run the full evaluation set:

```powershell
python scripts/08_evaluate.py --top-k 5 --batch-size 1 --batch-sleep 60 --max-retries 10 --timeout 400 --resume
```

Evidence retrieval runs sequentially because it shares one loaded SPLADE model.
Answer generation and RAGAS judging each run one sequential worker per Gemini
key, while all configured key workers run in parallel. Requests are paced for a safe
target of 13 RPM per key. Each key processes one question, sleeps for 60
seconds, and then receives its next question. The evaluator retries rate limits,
timeouts, and transient connection failures, checkpoints after every scored
question, and stores results in `data/eval/results.json`. If a RAGAS metric
returns NaN, that per-question metric is written as `null` and excluded from the
average. Custom sets of 18, 31, or more questions use the same JSON schema and
can be selected with `--questions`.

Experiments:

```text
dense only vs dense + SPLADE
without reranker vs with reranker
flat chunking vs parent-child chunking
with HyDE vs without HyDE
domain-filtered vs all-domain retrieval
```

## Environment Variables

Create a `.env` file locally.

```env
LLAMA_CLOUD_API_KEY=
FIRECRAWL_API_KEY=
VOYAGE_API_KEY=
GEMINI_API_KEY=
DATABASE_URL=

VOYAGE_EMBED_MODEL=voyage-multimodal-3.5
VOYAGE_RERANK_MODEL=rerank-2.5
GEMINI_MODEL=gemini-3.1-flash-lite
SPLADE_MODEL=naver/splade-cocondenser-ensembledistil
```

Use `.env.example` as the safe public template.

Do not commit `.env`.

## Local PostgreSQL + pgvector

This project uses the `pgvector/pgvector:pg16` Docker image. PostgreSQL data is
persisted in the Docker named volume `finalrag_postgres_data`.

Add the local database settings from `.env.example` to `.env`, then start and
initialize PostgreSQL:

```powershell
docker compose up -d db
docker compose ps
python scripts/01_create_database.py
```

The Python code connects through `DATABASE_URL`, so the repository layer stays
portable if the database is moved later.

Useful commands:

```powershell
# Stop PostgreSQL while preserving data
docker compose stop db

# Restart PostgreSQL
docker compose start db

# Follow database logs
docker compose logs -f db

# Back up the database
docker compose exec -T db pg_dump -U finalrag -d finalrag -Fc > finalrag.dump
```

Do not run `docker compose down -v` unless you intentionally want to delete the
database volume.

## Retrieval Commands

Rebuild domain centroids after adding or replacing dense embeddings:

```powershell
python scripts/06_index_chunks.py --skip-store --domain-centroids
```

Run the complete original-query retrieval and grounded-answer pipeline:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?"
```

For lower latency, use fast mode:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --fast
```

Fast mode uses smaller dense, sparse, fused, and rerank limits. It still uses
Voyage rerank-2.5. Full-quality mode remains the default.

Inspect retrieval results without calling Gemini:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --retrieval-only
```

Print complete parent and matched-child retrieval contexts:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --retrieval-only --full
```

The query path is:

```text
original query
-> domain centroid routing
-> Voyage dense retrieval + SPLADE retrieval
-> reciprocal rank fusion
-> child-to-parent expansion
-> deduplicate parents
-> Voyage rerank-2.5
-> Gemini grounded answer with numbered citations
```

## Build Order

```text
1. Create local project structure
2. Add .gitignore, .env.example, AGENTS.md, README.md
3. Start Docker PostgreSQL container
4. Enable pgvector extension
5. Create database tables
6. Add local files under data/input/{domain}/
7. Parse PDFs with LlamaParse
8. Parse HTML with Trafilatura / Firecrawl
9. Parse CSV with Pandas
10. Save raw parsed outputs locally
11. Normalize parser outputs into common elements
12. Create parent sections
13. Create child retrieval chunks
14. Generate Voyage dense embeddings
15. Generate SPLADE sparse vectors
16. Insert documents, parents, chunks, and vectors into PostgreSQL
17. Implement dense retrieval
18. Implement sparse retrieval
19. Add RRF fusion
20. Add child-to-parent expansion
21. Add Voyage reranking
22. Add Gemini grounded generation
23. Save query traces and citations
24. Run RAGAS evaluation
25. Compare retrieval experiments
```

## Current Scope

This version is local-only.

No frontend.

No FastAPI backend.

No authentication.

No deployment.

The interface is notebook or terminal based.

## Safety

Project files should remain inside this repository.

Source documents should be placed only inside:

```text
data/input/
```

Secrets should be stored only in `.env`.

Generated files, parsed private files, images, and local outputs should not be committed.

