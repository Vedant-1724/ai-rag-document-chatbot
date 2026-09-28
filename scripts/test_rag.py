import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve
from src.rag_pipeline import build_context
from src.generator import generate_answer


def main():
    question = (
        "What is Retrieval-Augmented Generation "
        "and how does it work?"
    )

    print("=" * 80)
    print("COMPLETE RAG PIPELINE TEST")
    print("=" * 80)
    print()

    print("QUESTION:")
    print(question)
    print()

    # ---------------------------------------------------------
    # STEP 1: RETRIEVAL
    # ---------------------------------------------------------

    print("=" * 80)
    print("STEP 1: RETRIEVING RELEVANT DOCUMENT CHUNKS")
    print("=" * 80)
    print()

    retrieved_chunks = retrieve(
        query=question,
        top_k=5,
    )

    if not retrieved_chunks:
        print("No relevant chunks were retrieved.")
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

    # ---------------------------------------------------------
    # STEP 2: BUILD CONTEXT
    # ---------------------------------------------------------

    print("=" * 80)
    print("STEP 2: BUILDING LLM CONTEXT")
    print("=" * 80)
    print()

    context = build_context(retrieved_chunks)

    print(f"Context built from {len(retrieved_chunks)} retrieved chunks.")
    print()

    # ---------------------------------------------------------
    # STEP 3: GENERATION
    # ---------------------------------------------------------

    print("=" * 80)
    print("STEP 3: GENERATING GROUNDED ANSWER")
    print("=" * 80)
    print()

    try:
        answer = generate_answer(
            question=question,
            context=context,
        )

        print("GENERATED ANSWER:")
        print()
        print(answer)

    except Exception as error:
        print("Gemini generation failed.")
        print()
        print(f"Error: {error}")
        print()
        print(
            "Retrieval was successful, but the LLM generation step "
            "was unavailable."
        )

    # ---------------------------------------------------------
    # STEP 4: SOURCES
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)
    print()

    for chunk in retrieved_chunks:
        print(
            f"- {chunk['source']} "
            f"(chunk {chunk['chunk_index']}, "
            f"distance {chunk['distance']:.4f})"
        )

    print()
    print("=" * 80)
    print("RAG PIPELINE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()