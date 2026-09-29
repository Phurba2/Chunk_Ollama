"""Ask questions about Markdown files using Ollama and PostgreSQL search."""

from __future__ import annotations

import sys

from ollama_rag import (
    DEFAULT_MODEL,
    markdown_rag_answer,
    test_ollama,
)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "What is EliteFreelancer?"

    if not test_ollama():
        raise SystemExit(1)

    result = markdown_rag_answer(
        question,
        model="llama3.2:3b",
        max_contexts=5,
    )

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for context in result["contexts"]:
        section = context["section_name"] or "Unknown section"
        print(
            f"- {context['filename']} "
            f"({section}, score={context['score']:.4f})"
        )

    print("\nPerformance:")
    print(f"- Contexts: {result['stats']['num_contexts']}")
    print(
        f"- Retrieval: "
        f"{result['stats']['retrieval_time_ms']:.1f} ms"
    )
    print(
        f"- Generation: "
        f"{result['stats']['generation_time_ms']:.1f} ms"
    )


if __name__ == "__main__":
    main()
