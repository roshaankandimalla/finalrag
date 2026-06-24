import csv
import json
import uuid
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import psycopg

from finalrag.database.repository import (
    clear_csv_data,
    insert_hcahps_records,
    upsert_hospital_category_docs,
    upsert_hospital_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CATEGORY_PREFIXES = {
    "nurse_communication": ("H_COMP_1", "H_NURSE"),
    "doctor_communication": ("H_COMP_2", "H_DOCTOR"),
    "medication_communication": ("H_COMP_5", "H_MED", "H_SIDE_EFFECTS"),
    "discharge_information": ("H_COMP_6", "H_DISCH", "H_SYMPTOMS"),
    "hospital_environment": ("H_CLEAN", "H_QUIET"),
    "overall_rating_and_recommendation": (
        "H_HSP",
        "H_RECMND",
        "H_STAR_RATING",
    ),
}

CATEGORY_LABELS = {
    "nurse_communication": "Nurse communication",
    "doctor_communication": "Doctor communication",
    "medication_communication": "Medication communication",
    "discharge_information": "Discharge information",
    "hospital_environment": "Hospital environment",
    "overall_rating_and_recommendation": "Overall rating and recommendation",
}

CSV_TO_DATABASE = {
    "Facility ID": "facility_id",
    "Facility Name": "facility_name",
    "Address": "address",
    "City/Town": "city",
    "State": "state",
    "ZIP Code": "zip_code",
    "County/Parish": "county",
    "Telephone Number": "telephone",
    "HCAHPS Measure ID": "measure_id",
    "HCAHPS Question": "question",
    "HCAHPS Answer Description": "answer_description",
    "Patient Survey Star Rating": "star_rating",
    "Patient Survey Star Rating Footnote": "star_rating_footnote",
    "HCAHPS Answer Percent": "answer_percent",
    "HCAHPS Answer Percent Footnote": "answer_percent_footnote",
    "HCAHPS Linear Mean Value": "linear_mean_value",
    "Number of Completed Surveys": "completed_surveys",
    "Number of Completed Surveys Footnote": "completed_surveys_footnote",
    "Survey Response Rate Percent": "response_rate_percent",
    "Survey Response Rate Percent Footnote": "response_rate_footnote",
    "Start Date": "survey_start_date",
    "End Date": "survey_end_date",
}


def clean_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    cleaned = str(value).strip()
    return cleaned or None


def parse_date(value) -> date | None:
    cleaned = clean_value(value)
    if not cleaned:
        return None
    return datetime.strptime(cleaned, "%m/%d/%Y").date()


def category_for_measure(measure_id: str) -> str:
    for category, prefixes in CATEGORY_PREFIXES.items():
        if measure_id.startswith(prefixes):
            return category
    raise ValueError(f"Unmapped HCAHPS measure ID: {measure_id}")


def row_result(row: dict) -> tuple[str, str]:
    candidates = [
        ("star rating", row.get("Patient Survey Star Rating")),
        ("answer percent", row.get("HCAHPS Answer Percent")),
        ("linear mean", row.get("HCAHPS Linear Mean Value")),
    ]

    for label, value in candidates:
        cleaned = clean_value(value)
        if cleaned and cleaned.lower() not in {
            "not applicable",
            "not available",
            "n/a",
        }:
            suffix = "%" if label == "answer percent" else ""
            return label, f"{cleaned}{suffix}"

    return "result", "Not available"


def markdown_cell(value) -> str:
    cleaned = clean_value(value) or ""
    return cleaned.replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def deterministic_uuid(*parts: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, ":".join(parts))


def raw_record(
    row: dict,
    document_id,
    source_row_number: int,
) -> dict:
    record = {
        database_name: clean_value(row.get(csv_name))
        for csv_name, database_name in CSV_TO_DATABASE.items()
    }
    record["document_id"] = document_id
    record["source_row_number"] = source_row_number
    record["survey_start_date"] = parse_date(row.get("Start Date"))
    record["survey_end_date"] = parse_date(row.get("End Date"))
    return record


def write_json_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, default=str))
    handle.write("\n")


def build_hospital_outputs(
    rows: list[dict],
    document_id,
    domain: str,
    file_name: str,
) -> tuple[dict, list[dict]]:
    first = rows[0]
    facility_id = clean_value(first["Facility ID"])
    profile_id = deterministic_uuid(str(document_id), facility_id, "profile")
    grouped_rows = defaultdict(list)

    for row in rows:
        grouped_rows[category_for_measure(row["HCAHPS Measure ID"])].append(row)

    category_summaries = {}
    category_docs = []

    for category in CATEGORY_PREFIXES:
        category_rows = grouped_rows.get(category, [])
        if not category_rows:
            continue

        summary = {}
        table_lines = [
            "| Measure ID | Measure | Result type | Result |",
            "|---|---|---|---:|",
        ]
        retrieval_lines = []
        measure_ids = []
        source_row_numbers = []

        for row in category_rows:
            measure_id = clean_value(row["HCAHPS Measure ID"])
            question = clean_value(row["HCAHPS Question"]) or measure_id
            result_type, result = row_result(row)
            source_row_number = int(row["_source_row_number"])

            summary[measure_id] = {
                "question": question,
                "result_type": result_type,
                "result": result,
            }
            table_lines.append(
                f"| {markdown_cell(measure_id)} | {markdown_cell(question)} | "
                f"{markdown_cell(result_type)} | {markdown_cell(result)} |"
            )
            retrieval_lines.append(f"- {question}: {result}")
            measure_ids.append(measure_id)
            source_row_numbers.append(source_row_number)

        category_summaries[category] = summary
        category_label = CATEGORY_LABELS[category]
        location = ", ".join(
            value
            for value in [
                clean_value(first["City/Town"]),
                clean_value(first["State"]),
            ]
            if value
        )
        retrieval_text = "\n".join(
            [
                f"Hospital: {clean_value(first['Facility Name'])}",
                f"Facility ID: {facility_id}",
                f"Location: {location}",
                f"Category: {category_label}",
                (
                    f"Survey period: {clean_value(first['Start Date'])} "
                    f"to {clean_value(first['End Date'])}"
                ),
                (
                    "Completed surveys: "
                    f"{clean_value(first['Number of Completed Surveys']) or 'Not available'}"
                ),
                (
                    "Survey response rate: "
                    f"{clean_value(first['Survey Response Rate Percent']) or 'Not available'}%"
                ),
                "",
                *retrieval_lines,
            ]
        )

        category_docs.append(
            {
                "category_doc_id": deterministic_uuid(
                    str(document_id), facility_id, category
                ),
                "profile_id": profile_id,
                "document_id": document_id,
                "facility_id": facility_id,
                "category": category,
                "retrieval_text": retrieval_text,
                "table_markdown": "\n".join(table_lines),
                "measure_ids": measure_ids,
                "source_row_numbers": source_row_numbers,
                "metadata": {
                    "domain": domain,
                    "source_type": "csv",
                    "file_name": file_name,
                    "hospital_name": clean_value(first["Facility Name"]),
                    "state": clean_value(first["State"]),
                    "survey_start_date": clean_value(first["Start Date"]),
                    "survey_end_date": clean_value(first["End Date"]),
                },
            }
        )

    category_overview = "\n".join(
        f"- {CATEGORY_LABELS[category]}: {len(summary)} measures"
        for category, summary in category_summaries.items()
    )
    profile_retrieval_text = "\n".join(
        [
            f"Hospital: {clean_value(first['Facility Name'])}",
            f"Facility ID: {facility_id}",
            (
                "Address: "
                f"{clean_value(first['Address'])}, "
                f"{clean_value(first['City/Town'])}, "
                f"{clean_value(first['State'])} "
                f"{clean_value(first['ZIP Code'])}"
            ),
            (
                f"Survey period: {clean_value(first['Start Date'])} "
                f"to {clean_value(first['End Date'])}"
            ),
            (
                "Completed surveys: "
                f"{clean_value(first['Number of Completed Surveys']) or 'Not available'}"
            ),
            (
                "Survey response rate: "
                f"{clean_value(first['Survey Response Rate Percent']) or 'Not available'}%"
            ),
            "",
            "Available HCAHPS categories:",
            category_overview,
        ]
    )

    profile = {
        "profile_id": profile_id,
        "document_id": document_id,
        "facility_id": facility_id,
        "hospital_name": clean_value(first["Facility Name"]),
        "address": clean_value(first["Address"]),
        "city": clean_value(first["City/Town"]),
        "state": clean_value(first["State"]),
        "zip_code": clean_value(first["ZIP Code"]),
        "county": clean_value(first["County/Parish"]),
        "telephone": clean_value(first["Telephone Number"]),
        "survey_start_date": parse_date(first["Start Date"]),
        "survey_end_date": parse_date(first["End Date"]),
        "completed_surveys": clean_value(first["Number of Completed Surveys"]),
        "response_rate_percent": clean_value(
            first["Survey Response Rate Percent"]
        ),
        "category_summaries": category_summaries,
        "retrieval_text": profile_retrieval_text,
        "metadata": {
            "domain": domain,
            "source_type": "csv",
            "file_name": file_name,
            "source_row_numbers": [
                int(row["_source_row_number"]) for row in rows
            ],
            "category_count": len(category_docs),
            "measure_count": len(rows),
        },
    }
    return profile, category_docs


def parse_hcahps_csv(
    connection: psycopg.Connection,
    document_id,
    file_path: str | Path,
    domain: str,
    batch_size: int = 5_000,
) -> dict:
    csv_path = Path(file_path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    parsed_dir = PROJECT_ROOT / "data" / "parsed" / domain / "csv"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = parsed_dir / f"{csv_path.stem}.metadata.json"
    profiles_path = parsed_dir / f"{csv_path.stem}.hospital_profiles.jsonl"
    category_docs_path = (
        parsed_dir / f"{csv_path.stem}.hospital_category_docs.jsonl"
    )

    clear_csv_data(connection, document_id)
    connection.commit()

    row_count = 0
    for frame in pd.read_csv(csv_path, dtype=str, chunksize=batch_size):
        records = []
        for row in frame.to_dict(orient="records"):
            row_count += 1
            records.append(raw_record(row, document_id, row_count + 1))

        insert_hcahps_records(connection, records)
        connection.commit()
        print(f"Inserted raw CSV rows: {row_count:,}")

    profile_count = 0
    category_doc_count = 0
    current_facility_id = None
    current_rows = []
    completed_facilities = set()

    with (
        csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file,
        profiles_path.open("w", encoding="utf-8") as profiles_file,
        category_docs_path.open("w", encoding="utf-8") as category_docs_file,
    ):
        reader = csv.DictReader(csv_file)

        for source_row_number, row in enumerate(reader, start=2):
            facility_id = clean_value(row["Facility ID"])
            row["_source_row_number"] = source_row_number

            if current_facility_id is None:
                current_facility_id = facility_id

            if facility_id != current_facility_id:
                if facility_id in completed_facilities:
                    raise RuntimeError(
                        "CSV rows are not grouped by Facility ID; streaming "
                        "hospital aggregation cannot continue safely."
                    )

                profile, category_docs = build_hospital_outputs(
                    current_rows,
                    document_id,
                    domain,
                    csv_path.name,
                )
                upsert_hospital_profile(connection, profile)
                upsert_hospital_category_docs(connection, category_docs)
                write_json_line(profiles_file, profile)
                for category_doc in category_docs:
                    write_json_line(category_docs_file, category_doc)

                completed_facilities.add(current_facility_id)
                profile_count += 1
                category_doc_count += len(category_docs)
                current_facility_id = facility_id
                current_rows = []

                if profile_count % 250 == 0:
                    connection.commit()
                    print(f"Created hospital profiles: {profile_count:,}")

            current_rows.append(row)

        if current_rows:
            profile, category_docs = build_hospital_outputs(
                current_rows,
                document_id,
                domain,
                csv_path.name,
            )
            upsert_hospital_profile(connection, profile)
            upsert_hospital_category_docs(connection, category_docs)
            write_json_line(profiles_file, profile)
            for category_doc in category_docs:
                write_json_line(category_docs_file, category_doc)
            profile_count += 1
            category_doc_count += len(category_docs)

    connection.commit()

    metadata = {
        "document_id": str(document_id),
        "domain": domain,
        "source_type": "csv",
        "file_name": csv_path.name,
        "row_count": row_count,
        "hospital_profile_count": profile_count,
        "hospital_category_doc_count": category_doc_count,
        "categories": list(CATEGORY_PREFIXES),
        "outputs": {
            "metadata_path": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
            "hospital_profiles_path": profiles_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "hospital_category_docs_path": category_docs_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata
