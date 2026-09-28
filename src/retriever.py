from src.embeddings import embed_text
from src.vector_store import search


DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = embed_text(query)

    results = search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for index, document in enumerate(documents):
        retrieved_chunks.append(
            {
                "chunk_id": ids[index],
                "content": document,
                "source": metadatas[index]["source"],
                "chunk_index": metadatas[index]["chunk_index"],
                "distance": distances[index],
            }
        )

    return retrieved_chunks