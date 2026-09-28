import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import generate_plain_answer, generate_answer
from src.retriever import retrieve
from src.rag_pipeline import build_context


def main():

    question = (
        "According to the Hugging Face RAG documentation, "
        "what does Retrieval-Augmented Generation combine, "
        "and how does it use retrieved passages during inference?"
    )

    print("=" * 80)
    print("WITHOUT RAG vs WITH RAG EXPERIMENT")
    print("=" * 80)
    print()

    print("QUESTION:")
    print(question)
    print()

    # =========================================================
    # 1. WITHOUT RAG
    # =========================================================

    print("=" * 80)
    print("1. WITHOUT RAG")
    print("=" * 80)
    print()

    try:

        print("Generating answer without document retrieval...")
        print()

        without_rag_answer = generate_plain_answer(question)

        print("ANSWER WITHOUT RAG:")
        print()
        print(without_rag_answer)
        print()

    except Exception as error:

        print("Without-RAG generation failed.")
        print()
        print(f"Error: {error}")
        print()

        without_rag_answer = (
            "[Without-RAG answer unavailable because "
            "the Gemini API was temporarily unavailable.]"
        )

    # =========================================================
    # 2. RETRIEVAL
    # =========================================================

    print("=" * 80)
    print("2. RETRIEVAL")
    print("=" * 80)
    print()

    print("Retrieving relevant document chunks...")
    print()

    retrieved_chunks = retrieve(
        query=question,
        top_k=5,
    )

    if not retrieved_chunks:
        print("No chunks were retrieved.")
        return

    for index, chunk in enumerate(retrieved_chunks, start=1):

        print("-" * 80)
        print(f"RESULT {index}")
        print("-" * 80)

        print(f"Source: {chunk['source']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Distance: {chunk['distance']:.4f}")

        print()
        print("CONTENT:")
        print(chunk["content"])
        print()

    # =========================================================
    # 3. WITH RAG
    # =========================================================

    print("=" * 80)
    print("3. WITH RAG")
    print("=" * 80)
    print()

    context = build_context(retrieved_chunks)

    try:

        print("Generating answer using retrieved document context...")
        print()

        with_rag_answer = generate_answer(
            question=question,
            context=context,
        )

        print("ANSWER WITH RAG:")
        print()
        print(with_rag_answer)
        print()

    except Exception as error:

        print("With-RAG generation failed.")
        print()
        print(f"Error: {error}")
        print()

        with_rag_answer = (
            "[With-RAG answer unavailable because "
            "the Gemini API was temporarily unavailable.]"
        )

    # =========================================================
    # 4. COMPARISON
    # =========================================================

    print("=" * 80)
    print("4. COMPARISON")
    print("=" * 80)
    print()

    print("WITHOUT RAG:")
    print()
    print(without_rag_answer)
    print()

    print("-" * 80)
    print()

    print("WITH RAG:")
    print()
    print(with_rag_answer)
    print()

    print("-" * 80)
    print()

    print("RETRIEVED SOURCES:")
    print()

    for chunk in retrieved_chunks:

        print(
            f"- {chunk['source']} "
            f"(chunk {chunk['chunk_index']}, "
            f"distance {chunk['distance']:.4f})"
        )

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()