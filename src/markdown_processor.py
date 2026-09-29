from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, MARKDOWN_STORAGE_PATH


class MarkdownProcessor:
    """Register Markdown files supplied in the markdown folder."""

    def __init__(self, db_config: dict, storage_path: Optional[Path] = None):
        self.db_config = db_config
        self.storage_path = Path(storage_path or MARKDOWN_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def ingest(self):
        files = sorted(self.storage_path.rglob("*.md"))
        conn = self._get_connection()
        new = existing = 0
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                for path in files:
                    cursor.execute("""
                        INSERT INTO papers (filename, title, file_path)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (filename) DO UPDATE SET
                            title = EXCLUDED.title,
                            file_path = EXCLUDED.file_path,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING (xmax = 0) AS inserted
                    """, (path.name, path.stem.replace("_", " "), str(path)))
                    if cursor.fetchone()["inserted"]:
                        new += 1
                    else:
                        existing += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {"found": len(files), "new": new, "existing": existing, "failed": 0}


def create_default_processor():
    return MarkdownProcessor({
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
    })
