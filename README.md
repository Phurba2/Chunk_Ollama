# Markdown Semantic Search with PostgreSQL, pgvector, and Ollama

This project searches your own Markdown (`.md`) files using natural-language questions, and answers them with a local Llama model.

It reads Markdown files from `markdown/`, splits their text into overlapping chunks, creates vector embeddings with `all-MiniLM-L6-v2`, and stores everything in PostgreSQL with `pgvector`. Questions are answered with hybrid search plus `llama3.2:3b` running locally through Ollama.

## 1. Install

```bash
git clone https://github.com/Phurba2/Chunk_Ollama.git
cd Chunk_Ollama
python3 -m venv env
source env/bin/activate

# Confirm Python and pip belong to this virtual environment
which python
which pip

# Both paths should contain: .../Chunk_Ollama/env/bin/
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

Only `.md` files are read.

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

Ask a question with the included CLI:

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

Each result carries the matched chunks:

```python
for result in search("your question"):
    print(f"File: {result.filename}")
    print(f"Score: {result.score:.4f}")
    for chunk in result.matched_chunks:
        print(chunk["section_name"], chunk["text"])
```

## 7. Ask questions with Ollama

The project includes a local RAG pipeline that retrieves Markdown chunks with hybrid search and answers with a local Llama model.

Install Ollama from [ollama.com](https://ollama.com), then in one terminal:

```bash
ollama serve
```

In another terminal, pull the model:

```bash
ollama pull llama3.2:3b
```

Ask a question:

```bash
python ask_ollama.py "Who runs the company?"
```

Example output:

```text
Answer:
Avi Rai runs the company as its CEO and co-founder, responsible for market strategy (a.md).

Sources:
- a.md (Who's Running the Show?, score=0.6193)

Performance:
- Contexts: 3
- Retrieval: 7163.0 ms
- Generation: 1636.1 ms
```

The pipeline lives in `ollama_rag.py`:

- `retrieve_contexts()` — hybrid search over `markdown/` chunks
- `format_rag_prompt()` — grounded prompt that forbids outside knowledge
- `call_ollama()` — non-streaming request to `http://localhost:11434`
- `markdown_rag_answer()` — retrieval + generation with timing stats

The model and URL are configured at the top of `ollama_rag.py`:

```python
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
```

## 8. Check the database

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
├── ask.py                      # CLI: print matching chunks
├── ask_ollama.py               # CLI: RAG answers via Ollama
├── ollama_rag.py               # Retrieval, prompting, Ollama client
├── setup_db.py                 # Initialize PostgreSQL tables
├── schema.sql                  # Database schema and indexes
├── requirements.txt
├── config/settings.py
└── src/
    ├── markdown_processor.py   # Register .md files
    ├── markdown_extractor.py   # Read headings and text
    ├── text_chunker.py         # Create overlapping chunks
    ├── embeddings.py           # all-MiniLM-L6-v2 embeddings
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

### `Unable to connect to Ollama`

Start the Ollama server in another terminal:

```bash
ollama serve
```

Then verify the model is installed:

```bash
ollama list
```

### `model '...' not found`

Pull the model named in `DEFAULT_MODEL`:

```bash
ollama pull llama3.2:3b
```

### Slow first question

The first search loads the embedding model, which adds several seconds. Later questions are much faster.
