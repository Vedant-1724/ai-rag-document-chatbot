import uuid

from src.embeddings import embed_texts
from src.vector_store import add_chunks, search


def test_vector_store_search():
    suffix = uuid.uuid4().hex

    chunks = [
        {
            "chunk_id": f"pytest::{suffix}::rag",
            "source": "pytest_document.md",
            "chunk_index": 0,
            "content": (
                "Retrieval-Augmented Generation combines "
                "retrieval with language model generation."
            ),
        },
        {
            "chunk_id": f"pytest::{suffix}::python",
            "source": "pytest_document.md",
            "chunk_index": 1,
            "content": (
                "Python is a programming language used "
                "for software development."
            ),
        },
    ]

    embeddings = embed_texts(
        [chunk["content"] for chunk in chunks],
        batch_size=2,
    )

    add_chunks(chunks, embeddings)

    query_embedding = embed_texts(
        ["How does retrieval augmented generation work?"]
    )[0]

    results = search(
        query_embedding=query_embedding,
        top_k=2,
    )

    assert len(results["documents"][0]) == 2

    assert (
        "Retrieval-Augmented Generation"
        in results["documents"][0][0]
    )