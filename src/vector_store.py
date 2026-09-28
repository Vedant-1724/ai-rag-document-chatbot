from pathlib import Path

import chromadb


COLLECTION_NAME = "rag_documents"
CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"


_client = None
_collection = None


def get_chroma_client():
    """Create or return the persistent ChromaDB client."""
    global _client

    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return _client


def get_collection():
    """Create or return the RAG document collection."""
    global _collection

    if _collection is None:
        client = get_chroma_client()

        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Document chunks for the RAG chatbot"
            },
        )

    return _collection


def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Add document chunks and their embeddings to ChromaDB."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings."
        )

    if not chunks:
        return

    collection = get_collection()

    ids = [chunk["chunk_id"] for chunk in chunks]

    documents = [chunk["content"] for chunk in chunks]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def search(
    query_embedding: list[float],
    top_k: int = 5,
) -> dict:
    """Search ChromaDB for the most similar document chunks."""

    collection = get_collection()

    if collection.count() == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )


def get_collection_count() -> int:
    """Return the number of stored chunks."""
    return get_collection().count()