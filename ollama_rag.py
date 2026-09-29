"""Local Ollama RAG integration for the Markdown search project."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from index import search
from src.search import SearchMode


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


@dataclass
class RetrievedContext:
    filename: str
    title: str
    section_name: str | None
    text: str
    score: float


def _ollama_request(
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Send a JSON request to Ollama."""
    url = f"{OLLAMA_URL}{endpoint}"

    request = Request(
        url,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
    )

    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> str:
    """Generate a response using a local Ollama model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    try:
        result = _ollama_request(
            "/api/generate",
            payload=payload,
            timeout=timeout,
        )
        return result.get("response", "").strip()

    except HTTPError as error:
        return f"Ollama HTTP error {error.code}: {error.reason}"

    except URLError:
        return (
            "Unable to connect to Ollama. "
            "Start it with: ollama serve"
        )

    except Exception as error:
        return f"Error calling Ollama: {error}"


def test_ollama(verbose: bool = True) -> bool:
    """Check whether Ollama is available and list installed models."""
    try:
        result = _ollama_request("/api/tags", timeout=5)
        models = result.get("models", [])
        names = [model.get("name") for model in models]

        if verbose:
            print("Available Ollama models:")
            for name in names:
                print(f"  - {name}")

        return True

    except Exception:
        if verbose:
            print("Ollama is not running. Start it with: ollama serve")
        return False


def retrieve_contexts(
    question: str,
    max_contexts: int = 5,
    max_chars_per_context: int = 500,
) -> list[RetrievedContext]:
    """Retrieve Markdown chunks using the project's hybrid search."""
    results = search(
        question,
        mode=SearchMode.HYBRID,
        limit=max_contexts,
    )

    contexts: list[RetrievedContext] = []

    for result in results:
        for chunk in result.matched_chunks:
            text = chunk["text"].strip()

            if not text:
                continue

            contexts.append(
                RetrievedContext(
                    filename=result.filename,
                    title=result.title,
                    section_name=chunk.get("section_name"),
                    text=text[:max_chars_per_context],
                    score=float(chunk.get("score", result.score)),
                )
            )

            if len(contexts) >= max_contexts:
                return contexts

    return contexts


def format_rag_prompt(
    question: str,
    contexts: list[RetrievedContext],
) -> str:
    """Build a grounded prompt for Llama using retrieved Markdown context."""

    if not contexts:
        return f"""You are a document question-answering assistant.

Question:
{question}

No relevant context was found. Reply exactly:
I could not find that information in the indexed documents.
"""

    context_parts = []

    for number, context in enumerate(contexts, start=1):
        section = context.section_name or "Unknown section"

        context_parts.append(
            f"""Source {number}
File: {context.filename}
Section: {section}
Score: {context.score:.4f}

{context.text}"""
        )

    context_text = "\n\n".join(context_parts)

    return f"""You are a document question-answering assistant.

Use all relevant retrieved sections to answer the question.

Rules:
- Return only the final answer.
- Use clean Markdown formatting.
- Do not include a Sources section.
- Do not include similarity scores.
- Do not mention retrieved context.
- Do not mention filenames or section names.
- Combine all relevant information from the supplied sections.
- Use headings and bullet points when they improve readability.
- Do not use outside knowledge.
- If the answer is not present in the context, say:
  "I could not find that information in the indexed documents."
- The retrieved sections almost always contain the answer; the fallback
  above is only for questions completely unrelated to the content.

Question:
{question}

Retrieved document sections:
{context_text}

The sections above do contain the answer to the question. Answer it now.

Final answer only:"""


def markdown_rag_answer(
    question: str,
    model: str = DEFAULT_MODEL,
    max_contexts: int = 5,
) -> dict[str, Any]:
    """Retrieve Markdown context and generate an Ollama answer."""
    retrieval_start = time.perf_counter()

    contexts = retrieve_contexts(
        question,
        max_contexts=max_contexts,
    )

    retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

    if not contexts:
        return {
            "answer": (
                "I could not find relevant information "
                "in the indexed Markdown documents."
            ),
            "contexts": [],
            "stats": {
                "retrieval_time_ms": round(retrieval_time_ms, 1),
                "generation_time_ms": 0.0,
                "num_contexts": 0,
            },
        }

    prompt = format_rag_prompt(question, contexts)

    generation_start = time.perf_counter()
    answer = call_ollama(prompt, model=model)
    generation_time_ms = (time.perf_counter() - generation_start) * 1000

    return {
        "answer": answer,
        "contexts": [asdict(context) for context in contexts],
        "stats": {
            "retrieval_time_ms": round(retrieval_time_ms, 1),
            "generation_time_ms": round(generation_time_ms, 1),
            "num_contexts": len(contexts),
        },
    }
