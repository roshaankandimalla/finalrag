import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# This script is the single production parsing entry point.
# The parser implementations stay in src/finalrag/parsing; this file only
# coordinates source-specific behavior and database status updates.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from finalrag.database.repository import (
    fetch_documents_for_parsing,
    mark_document_failed,
    mark_document_parsed,
    save_llamaparse_job,
    update_document_status,
)
from finalrag.parsing.csv_parser import parse_hcahps_csv
from finalrag.parsing.html_parser import (
    find_existing_html_outputs,
    parse_html_crawl,
)
from finalrag.parsing.pdf_parser import find_existing_pdf_outputs, parse_pdf


PARSER_GROUPS = ("pdf", "html", "csv")


# ---------------------------------------------------------------------------
# CLI and shared database helpers
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse discovered PDF, HTML, and CSV sources."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=PARSER_GROUPS,
        help="Run only selected parser groups, for example: --only pdf html",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=PARSER_GROUPS,
        default=[],
        help="Skip selected parser groups, for example: --skip csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pending parse work without external jobs or database updates.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining parser groups even if one parser group fails.",
    )
    return parser.parse_args()


def selected_groups(args: argparse.Namespace) -> list[str]:
    groups = list(PARSER_GROUPS)

    if args.only:
        groups = [group for group in groups if group in args.only]

    if args.skip:
        groups = [group for group in groups if group not in args.skip]

    return groups


def require_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")
    return database_url


def load_llamaparse_api_keys() -> dict[str, str]:
    keys = {}

    for number in range(1, 4):
        alias = f"key_{number}"
        value = os.environ.get(f"LLAMA_CLOUD_API_KEY_{number}")
        if value:
            keys[alias] = value

    if not keys:
        raise RuntimeError(
            "Add LLAMA_CLOUD_API_KEY_1, LLAMA_CLOUD_API_KEY_2, and/or "
            "LLAMA_CLOUD_API_KEY_3 to .env"
        )

    return keys


def existing_llamaparse_job(document: dict) -> tuple[str | None, str | None]:
    llamaparse = (document.get("metadata") or {}).get("llamaparse") or {}
    return llamaparse.get("job_id"), llamaparse.get("api_key_alias")


def update_status(database_url: str, document_id, status: str) -> None:
    with psycopg.connect(database_url) as connection:
        update_document_status(connection, document_id, status)


def save_llamaparse_job_id(
    database_url: str,
    document_id,
    job_id: str,
    api_key_alias: str,
) -> None:
    with psycopg.connect(database_url) as connection:
        save_llamaparse_job(connection, document_id, job_id, api_key_alias)


def save_parsed_outputs(database_url: str, document_id, outputs: dict) -> None:
    with psycopg.connect(database_url) as connection:
        mark_document_parsed(connection, document_id, outputs)


def save_failure(database_url: str, document_id, error: Exception | str) -> None:
    with psycopg.connect(database_url) as connection:
        mark_document_failed(connection, document_id, error)


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------


def assign_pdf_documents(
    documents: list[dict],
    api_keys: dict[str, str],
) -> list[tuple[dict, str, str]]:
    aliases = list(api_keys)
    assignments = []

    for index, document in enumerate(documents):
        _, existing_alias = existing_llamaparse_job(document)
        alias = existing_alias or aliases[index % len(aliases)]

        if alias not in api_keys:
            raise RuntimeError(
                f"{document['file_name']} uses stored API key alias "
                f"{alias!r}, but that key is not available in .env"
            )

        assignments.append((document, alias, api_keys[alias]))

    return assignments


def parse_one_pdf(
    document: dict,
    api_key_alias: str,
    api_key: str,
    database_url: str,
) -> dict:
    document_id = document["document_id"]
    existing_job_id, _ = existing_llamaparse_job(document)

    # If a previous LlamaParse run already wrote all local outputs, reuse those
    # files and only update PostgreSQL status. This prevents spending credits
    # for the same PDF again.
    existing_outputs = find_existing_pdf_outputs(
        document["file_path"],
        document["domain"],
    )
    if existing_outputs:
        if existing_job_id:
            existing_outputs["job_id"] = existing_job_id

        save_parsed_outputs(database_url, document_id, existing_outputs)
        return {
            "file_name": document["file_name"],
            "status": "reused",
            "job_id": existing_job_id or "existing-output",
            "image_count": existing_outputs["downloaded_image_count"],
        }

    update_status(database_url, document_id, "parsing")

    def save_new_job(job_id: str) -> None:
        save_llamaparse_job_id(database_url, document_id, job_id, api_key_alias)

    try:
        outputs = parse_pdf(
            file_path=document["file_path"],
            domain=document["domain"],
            api_key=api_key,
            existing_job_id=existing_job_id,
            on_job_created=save_new_job,
        )
        save_parsed_outputs(database_url, document_id, outputs)

        return {
            "file_name": document["file_name"],
            "status": "parsed",
            "job_id": outputs["job_id"],
            "image_count": outputs["downloaded_image_count"],
        }
    except Exception as exc:
        try:
            save_failure(database_url, document_id, exc)
        except Exception as database_exc:
            print(
                f"Could not save failure status for {document['file_name']}: "
                f"{database_exc}"
            )
        raise


def parse_pdf_key_queue(
    assignments: list[tuple[dict, str, str]],
    database_url: str,
) -> list[dict]:
    # Each API key processes its assigned PDFs sequentially. Different keys run
    # in parallel, but one key never starts two jobs at the same time.
    results = []

    for document, alias, api_key in assignments:
        try:
            result = parse_one_pdf(document, alias, api_key, database_url)
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "file_name": document["file_name"],
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return results


def parse_pdfs(database_url: str, dry_run: bool) -> int:
    with psycopg.connect(database_url) as connection:
        documents = fetch_documents_for_parsing(connection, source_type="pdf")

    if not documents:
        print("No discovered or failed PDFs need parsing.")
        return 0

    api_keys = load_llamaparse_api_keys()
    assignments = assign_pdf_documents(documents, api_keys)

    print(f"PDFs to parse: {len(assignments)}")
    for document, alias, _ in assignments:
        job_id, _ = existing_llamaparse_job(document)
        existing_outputs = find_existing_pdf_outputs(
            document["file_path"],
            document["domain"],
        )
        if existing_outputs:
            action = "reuse existing local outputs"
        elif job_id:
            action = f"resume {job_id}"
        else:
            action = "create new job"
        print(f"  {document['file_name']} -> {alias} ({action})")

    if dry_run:
        print("PDF dry-run complete. No jobs were created.")
        return 0

    # Split the work by key alias so one slow PDF does not block the other API
    # keys, while still preserving the per-key sequential queue.
    queues = {
        alias: [assignment for assignment in assignments if assignment[1] == alias]
        for alias in api_keys
    }
    queues = {alias: queue for alias, queue in queues.items() if queue}

    results = []
    with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
        futures = {
            executor.submit(parse_pdf_key_queue, queue, database_url): alias
            for alias, queue in queues.items()
        }

        for future in as_completed(futures):
            alias = futures[future]
            key_results = future.result()
            results.extend(key_results)

            for result in key_results:
                file_name = result["file_name"]
                if result["status"] != "failed":
                    print(
                        f"Parsed: {file_name} | key={alias} | "
                        f"job={result['job_id']} | "
                        f"images={result['image_count']}"
                    )
                else:
                    print(f"Failed: {file_name} | key={alias} | {result['error']}")

    failures = [result for result in results if result["status"] == "failed"]
    print(f"\nPDF completed: {len(assignments) - len(failures)}")
    print(f"PDF failed:    {len(failures)}")

    if failures:
        failed_names = ", ".join(result["file_name"] for result in failures)
        raise RuntimeError(
            f"Some PDFs failed: {failed_names}. Re-run this script to resume "
            "them from their saved LlamaParse job IDs."
        )

    return len(assignments)


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


def parse_html_sources(database_url: str, dry_run: bool) -> int:
    with psycopg.connect(database_url) as connection:
        documents = fetch_documents_for_parsing(connection, source_type="html")

    if not documents:
        print("No discovered, parsing, or failed HTML sources need parsing.")
        return 0

    print(f"HTML sources to parse: {len(documents)}")
    needs_firecrawl = False
    for document in documents:
        # Firecrawl is skipped when local parsed outputs already exist.
        existing = find_existing_html_outputs(
            document["file_name"],
            document["domain"],
        )
        if existing:
            action = "reuse existing local outputs"
        else:
            action = "start Firecrawl crawl"
            needs_firecrawl = True
        print(f"  {document['file_name']} -> {action}")

    if dry_run:
        print("HTML dry-run complete. No crawls were started.")
        return 0

    firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")
    if needs_firecrawl and not firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is missing from .env")

    failures = []
    for document in documents:
        name = document["file_name"]
        domain = document["domain"]
        config = (document.get("metadata") or {}).get("firecrawl_config") or {}
        print(f"\nHTML source: {name}")

        try:
            existing = find_existing_html_outputs(name, domain)
            if existing:
                outputs = existing
                print(f"[{name}] Reusing existing local outputs")
            else:
                update_status(database_url, document["document_id"], "parsing")
                outputs = parse_html_crawl(
                    name=name,
                    domain=domain,
                    config=config,
                    api_key=firecrawl_api_key,
                )

            save_parsed_outputs(database_url, document["document_id"], outputs)
            print(
                f"Parsed: {name} | pages={outputs['page_count']} | "
                f"sections={outputs['section_count']}"
            )
        except Exception as exc:
            failures.append((name, str(exc)))
            try:
                save_failure(database_url, document["document_id"], exc)
            except Exception as database_exc:
                print(f"Could not save failure status: {database_exc}")
            print(f"Failed: {name} | {exc}")

    print(f"\nHTML completed: {len(documents) - len(failures)}")
    print(f"HTML failed:    {len(failures)}")

    if failures:
        raise RuntimeError(
            "Some HTML sources failed: "
            + ", ".join(name for name, _ in failures)
        )

    return len(documents)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_csv_documents(database_url: str, dry_run: bool) -> int:
    with psycopg.connect(database_url) as connection:
        documents = fetch_documents_for_parsing(connection, source_type="csv")

    if not documents:
        print("No discovered, parsing, or failed CSV documents need parsing.")
        return 0

    print(f"CSV documents to parse: {len(documents)}")
    for document in documents:
        print(f"  {document['file_name']} -> parse with pandas")

    if dry_run:
        print("CSV dry-run complete. No rows were parsed or stored.")
        return 0

    for document in documents:
        print(f"\nParsing CSV: {document['file_name']}")

        try:
            with psycopg.connect(database_url) as connection:
                # CSV parsing stores raw records plus derived hospital profiles
                # and category documents in one transaction.
                update_document_status(
                    connection,
                    document["document_id"],
                    "parsing",
                )
                connection.commit()

                outputs = parse_hcahps_csv(
                    connection=connection,
                    document_id=document["document_id"],
                    file_path=document["file_path"],
                    domain=document["domain"],
                )

                mark_document_parsed(
                    connection,
                    document["document_id"],
                    outputs,
                )
                connection.commit()

            print(
                f"Parsed: {document['file_name']} | "
                f"rows={outputs['row_count']:,} | "
                f"hospitals={outputs['hospital_profile_count']:,} | "
                f"category_docs={outputs['hospital_category_doc_count']:,}"
            )
        except Exception as exc:
            with psycopg.connect(database_url) as connection:
                mark_document_failed(connection, document["document_id"], exc)
                connection.commit()
            print(f"Failed: {document['file_name']} | {exc}")
            raise

    return len(documents)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_group(group: str, database_url: str, dry_run: bool) -> int:
    print(f"\n{'=' * 72}")
    print(f"{group.upper()} parsing")
    print(f"{'=' * 72}")

    if group == "pdf":
        return parse_pdfs(database_url, dry_run)
    if group == "html":
        return parse_html_sources(database_url, dry_run)
    if group == "csv":
        return parse_csv_documents(database_url, dry_run)

    raise ValueError(f"Unknown parser group: {group}")


def main() -> None:
    args = parse_arguments()
    groups = selected_groups(args)

    if not groups:
        print("No parser groups selected.")
        return

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    database_url = require_database_url()

    failures = []
    for group in groups:
        try:
            run_group(group, database_url, args.dry_run)
        except Exception as exc:
            failures.append((group, str(exc)))
            print(f"\n{group.upper()} parsing failed: {exc}")
            if not args.continue_on_error:
                raise

    print(f"\nParser groups completed: {len(groups) - len(failures)}")
    print(f"Parser groups failed:    {len(failures)}")

    if failures:
        raise RuntimeError(
            "Parser groups failed: "
            + ", ".join(group for group, _ in failures)
        )


if __name__ == "__main__":
    main()
