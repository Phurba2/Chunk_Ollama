import sys

from index import search


question = " ".join(sys.argv[1:]) or "What is EliteFreelancer?"

for result in search(question):
    print(f"File: {result.filename}")
    print(f"Score: {result.score:.4f}")

    for number, chunk in enumerate(result.matched_chunks, start=1):
        print(f"\n--- Match {number} ---")
        print(chunk["text"])
