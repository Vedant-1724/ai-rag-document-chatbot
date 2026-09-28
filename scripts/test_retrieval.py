import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve


def main():
    query = (
        "How does Retrieval-Augmented Generation "
        "combine retrieval with language model generation?"
    )

    print("=" * 80)
    print("RAG RETRIEVAL TEST")
    print("=" * 80)

    print()
    print(f"Query: {query}")
    print()

    results = retrieve(query, top_k=5)

    print("=" * 80)
    print(f"RETRIEVED {len(results)} RELEVANT CHUNKS")
    print("=" * 80)

    for index, result in enumerate(results, start=1):
        print()
        print("-" * 80)
        print(f"RESULT {index}")
        print("-" * 80)

        print(f"Source: {result['source']}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Distance: {result['distance']}")
        print()

        print("CONTENT:")
        print(result["content"])


if __name__ == "__main__":
    main()