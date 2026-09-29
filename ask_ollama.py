from __future__ import annotations

import sys

from ollama_rag import markdown_rag_answer, test_ollama


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "Who runs the company?"

    if not test_ollama(verbose=False):
        print("Ollama is not running. Start it with: ollama serve")
        raise SystemExit(1)

    result = markdown_rag_answer(
        question,
        model="llama3.2:3b",
        max_contexts=5,
    )

    print(result["answer"].strip())


if __name__ == "__main__":
    main()
