import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "src/finalrag/database/schema.sql"

load_dotenv(PROJECT_ROOT / ".env", override=True)

database_url = os.environ["DATABASE_URL"]
schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

print(f"Schema file: {SCHEMA_PATH}")
print(f"Schema characters: {len(schema_sql)}")

if not schema_sql.strip():
    raise RuntimeError("schema.sql is empty")

# Execute and explicitly commit
with psycopg.connect(database_url) as connection:
    with connection.cursor() as cursor:
        cursor.execute(schema_sql)

    connection.commit()

print("Database schema committed successfully.")

# Verify after commit
with psycopg.connect(database_url) as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database(), current_schema();")
        print("Connected to:", cursor.fetchone())

        cursor.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)

        print("Tables:", cursor.fetchall())
