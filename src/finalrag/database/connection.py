import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    return database_url


def connect_database() -> psycopg.Connection:
    connection = psycopg.connect(get_database_url())
    register_vector(connection)
    return connection
