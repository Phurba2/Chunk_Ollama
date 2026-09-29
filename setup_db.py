"""Initialize the configured PostgreSQL database schema."""

from pathlib import Path

import psycopg2

from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


schema = (Path(__file__).parent / "schema.sql").read_text()

with psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(schema)

print(f"Schema initialized in database: {DB_NAME}")
