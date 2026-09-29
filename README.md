# Markdown Semantic Search with PostgreSQL and pgvector

This project searches your own Markdown (`.md`) files using natural-language questions.

It reads Markdown files from `markdown/`, splits their text into overlapping chunks, creates vector embeddings with `all-MiniLM-L6-v2`, and stores everything in PostgreSQL with `pgvector`.

## 1. Install

```bash
git clone https://github.com/Phurba2/.md_Search.git
cd .md_search
python3 -m venv env
source env/bin/activate

# Confirm Python and pip belong to this virtual environment
which python
which pip

# Both paths should contain: /MarkDown_similarity_search_using_postgres/env/bin/
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Configure PostgreSQL

Create `.env` in the project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=md_vector
DB_USER=furba
DB_PASSWORD=furba
```

Create the database if necessary:

```bash
sudo -u postgres createdb md_vector
```

Enable the required extensions:

```bash
sudo -u postgres psql -d md_vector -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d md_vector -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

Create the tables:

```bash
python setup_db.py
```

## 3. Add Markdown files

Put your files in `markdown/`:

```text
markdown/
├── money.md
└── machine_learning.md
```

Only `.md` files are read. The folder is tracked in Git, but Markdown documents are ignored so private files are not uploaded.

## 4. Index the Markdown files

Run:

```bash
python - <<'PY'
from index import ingest_and_embed
print(ingest_and_embed())
PY
```

This registers each Markdown file, reads it as UTF-8, splits it into chunks, generates embeddings, and stores the chunks and vectors in PostgreSQL. The first run downloads the `all-MiniLM-L6-v2` model.

## 5. How chunking works

Markdown headings are kept as section names. For example:

```md
# Psychology of Money

## Staying Wealthy

Staying wealthy requires avoiding ruin and surviving difficult periods.
```

The heading becomes metadata such as `section_name = 'Staying Wealthy'`. Text is then grouped into chunks of approximately 768 words, with a maximum of about 1,024 words and roughly 128 words of overlap between neighboring chunks. Overlap preserves context when an idea crosses a chunk boundary.

## 6. Search

Create `ask.py` in the project root (the repository includes the same example):

```python
import sys
from index import search

question = " ".join(sys.argv[1:]) or "What is EliteFreelancer?"

for result in search(question):
    print(f"File: {result.filename}")
    print(f"Score: {result.score:.4f}")
    for chunk in result.matched_chunks:
        print("\n--- Match ---")
        print(chunk["text"])
```

Ask a question:

```bash
python ask.py "What services does EliteFreelancer provide?"
```

The default is hybrid search. It combines vector similarity through `pgvector` with keyword similarity through PostgreSQL `pg_trgm`.

Use a specific mode when needed:

```python
from index import search
from src.search import SearchMode

vector_results = search("your question", mode=SearchMode.VECTOR)
keyword_results = search("your words", mode=SearchMode.KEYWORD)
hybrid_results = search("your question", mode=SearchMode.HYBRID)
```

## 7. Check the database

```bash
psql -h localhost -U furba -d md_vector -c "SELECT id, filename, processed, embedding_generated FROM papers;"
```

Check chunks and embeddings:

```bash
psql -h localhost -U furba -d md_vector -c "
SELECT p.filename, COUNT(c.id) AS chunks,
       COUNT(c.embedding) AS embeddings
FROM papers p
LEFT JOIN paper_chunks c ON c.paper_id = p.id
GROUP BY p.filename;
"
```

## Project structure

```text
.
├── markdown/                   # Put .md files here
├── index.py                    # ingest_and_embed() and search()
├── setup_db.py                 # Initialize PostgreSQL tables
├── schema.sql                  # Database schema and indexes
├── requirements.txt
├── config/settings.py
└── src/
    ├── markdown_processor.py   # Register .md files
    ├── markdown_extractor.py  # Read headings and text
    ├── text_chunker.py         # Create overlapping chunks
    ├── embeddings.py            # all-MiniLM-L6-v2 embeddings
    ├── embedding_pipeline.py   # Store chunks and vectors
    └── search.py               # Vector, keyword, and hybrid search
```

## Troubleshooting

### `type "vector" does not exist`

```bash
sudo -u postgres psql -d md_vector -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### No documents are found

Make sure files end in `.md` and are inside `markdown/`, then run indexing again.

### No text is extracted

Save the file as UTF-8 Markdown. Markdown is plain text and does not require OCR or a PDF parser.
