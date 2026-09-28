import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve


QUERIES = [
    "What is Retrieval-Augmented Generation and how does it work?",
    "How do embeddings improve retrieval augmented generation systems?",
    "What is a Python garbage collector and how does it work?",
    "What are the main components of a RAG system?",
    "How does Gemini generate text using the Gemini API?",
]


def main():
    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    for query_number, query in enumerate(QUERIES, start=1):

        print()
        print("=" * 80)
        print(f"QUERY {query_number}")
        print("=" * 80)
        print()
        print(query)

        results = retrieve(
            query=query,
            top_k=5,
        )

        print()
        print(f"Relevant chunks returned: {len(results)}")
        print()

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"Distance={result['distance']:.4f} | "
                f"Source={result['source']} | "
                f"Chunk={result['chunk_index']}"
            )


if __name__ == "__main__":
    main()