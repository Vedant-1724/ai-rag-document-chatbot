import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import answer_question


def print_sources(sources: list[dict]) -> None:
    print()
    print("-" * 80)
    print("SOURCES")
    print("-" * 80)

    if not sources:
        print("No sources found.")
        return

    for source in sources:
        print(
            f"- {source['source']} "
            f"(chunk {source['chunk_index']}, "
            f"distance {source['distance']:.4f})"
        )


def main():
    print("=" * 80)
    print("AI RAG DOCUMENT CHATBOT")
    print("=" * 80)
    print()
    print("Ask questions about the indexed document collection.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            print()
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print()
        print("Retrieving relevant documents...")
        print()

        try:
            result = answer_question(
                question=question,
                top_k=5,
            )

            print("Assistant:")
            print()
            print(result["answer"])

            print_sources(result.get("sources", []))

        except Exception as error:
            print()
            print("An error occurred:")
            print(error)

        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()