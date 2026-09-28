from src.generator import generate_answer
from src.retriever import retrieve


DEFAULT_TOP_K = 5


def build_context(retrieved_chunks: list[dict]) -> str:
    """
    Combine retrieved chunks into a context string for the LLM.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Context {index}]\n"
            f"Source: {chunk['source']}\n"
            f"Chunk: {chunk['chunk_index']}\n"
            f"{chunk['content']}"
        )

    return "\n\n".join(context_parts)


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Complete RAG pipeline:

    Question
        ↓
    Retrieval
        ↓
    Context construction
        ↓
    Gemini generation
    """

    retrieved_chunks = retrieve(
        query=question,
        top_k=top_k,
    )

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": (
                "I could not find relevant information "
                "in the document collection."
            ),
            "retrieved_chunks": [],
            "context": "",
            "sources": [],
        }

    context = build_context(retrieved_chunks)

    answer = generate_answer(
        question=question,
        context=context,
    )

    sources = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "distance": chunk["distance"],
        }
        for chunk in retrieved_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "sources": sources,
    }