# Multimodal Hybrid RAG Portfolio Project

This project is a local multimodal RAG system built over finance, legal, and medical documents. It supports PDF, HTML, and CSV sources, then turns them into a unified retrieval pipeline with dense multimodal embeddings, SPLADE sparse retrieval, parent-child chunking, reranking, grounded answer generation, citations, and RAGAS evaluation.

The goal is to demonstrate an enterprise-style RAG architecture, not just a simple vector search demo.

## What This Project Shows

- Domain-aware document ingestion for finance, legal, and medical files
- PDF parsing with LlamaParse
- HTML parsing with Firecrawl / Trafilatura
- Large CSV processing with Pandas
- Normalization into a common element schema
- Hierarchical parent-child chunking
- Multimodal child chunks with text, tables, and image paths
- Voyage multimodal dense embeddings
- SPLADE sparse vectors for exact lexical retrieval
- PostgreSQL + pgvector storage
- Hybrid retrieval with Reciprocal Rank Fusion
- Child-to-parent context expansion
- Voyage reranking
- Gemini grounded answer generation
- Citations with file, page, section, and chunk metadata
- RAGAS evaluation with saved metrics

## Architecture

```text
Local files
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
      -> RRF fusion
  -> child-to-parent expansion
  -> Voyage reranker
  -> Gemini answer generation
  -> grounded answer with citations
  -> RAGAS evaluation
```

## Why Multimodal

The pipeline keeps different document evidence types together:

- Text is used directly for retrieval and generation.
- Tables are represented as markdown for retrieval and HTML for generation.
- Images are stored as file paths and loaded only when needed for embedding or generation.
- Image base64 is created temporarily at generation time, not stored in the database.

This keeps storage efficient while still allowing multimodal reasoning.

## Database Design

The core tables are:

```text
documents
parent_sections
child_chunks
hospital_profiles
hospital_category_docs
queries
citations
```

`documents` stores one row per source document.

`parent_sections` stores larger context blocks.

`child_chunks` stores retrieval units and vectors.

For CSV data, hospital-level aggregate tables are used so the system does not need to reconstruct hospitals from raw measure rows every time.

## Retrieval Flow

```text
User query
  -> domain routing
  -> dense Voyage retrieval
  -> SPLADE sparse retrieval
  -> reciprocal rank fusion
  -> parent expansion
  -> deduplication
  -> Voyage reranking
  -> top contexts
  -> Gemini answer with citations
```

The system uses the original user query. Query rewriting is intentionally not required for the main pipeline, which keeps the retrieval behavior easier to inspect and evaluate.

## Example Query

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?"
```

Fast mode:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --fast
```

Retrieval-only debugging:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?" --retrieval-only
```

## Evaluation

RAGAS metrics are used to evaluate answer and retrieval quality:

```text
answer_relevancy
context_precision
context_recall
faithfulness
```

Selected evaluation files are kept in `data/eval/` so results can be inspected directly.

Example:

```powershell
python scripts/08_evaluate.py --questions data/eval/questions.json --output data/eval/results.json --top-k 8 --batch-size 1 --batch-sleep 60 --max-retries 10 --timeout 400 --resume
```

## Project Layout

```text
src/finalrag/
  discovery/       file discovery
  parsing/         PDF, HTML, CSV parsers
  normalization/   common element conversion
  chunking/        hierarchical chunking
  embeddings/      Voyage and SPLADE embedding code
  database/        PostgreSQL connection and repository layer
  retrieval/       dense, sparse, RRF, parent expansion, reranking
  generation/      context building, citations, Gemini generation
  evaluation/      RAGAS evaluation
  graphing/        unified graph tracing

scripts/
  01_create_database.py
  02_discover_files.py
  03_parse_all.py
  04_normalize_elements.py
  05_create_chunks.py
  06_index_chunks.py
  07_query.py
  08_evaluate.py

tests/
  unit tests for discovery, parsing, normalization, chunking, retrieval,
  embeddings, database helpers, citations, and graphing
```

## Setup

Create `.env` from `.env.example` and fill in local API keys.

Start PostgreSQL + pgvector:

```powershell
docker compose up -d db
python scripts/01_create_database.py
```

Run the pipeline:

```powershell
python scripts/02_discover_files.py
python scripts/03_parse_all.py
python scripts/04_normalize_elements.py
python scripts/05_create_chunks.py
python scripts/06_index_chunks.py
```

Ask a question:

```powershell
python scripts/07_query.py "What was Reliance Industries financial performance?"
```

Run tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- `.env` is never committed.
- Large generated artifacts and logs are ignored where appropriate.
- Some output examples and evaluation results are intentionally kept for project review.
- The project is local-first and does not include a frontend or backend API in this version.

## Why This Matters

This project demonstrates the practical engineering needed for real RAG systems:

- handling mixed document formats
- preserving source metadata
- combining semantic and lexical retrieval
- using parent-child context expansion
- grounding answers with citations
- evaluating quality with RAGAS
- keeping the pipeline reproducible through scripts and tests

