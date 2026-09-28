from src.embeddings import (
    embed_text,
    embed_texts,
    get_embedding_dimension,
)


def test_embedding_dimension():
    embedding = embed_text("Retrieval-Augmented Generation is useful.")

    assert len(embedding) == get_embedding_dimension()
    assert len(embedding) == 384


def test_multiple_embeddings():
    texts = [
        "Python is a programming language.",
        "Machine learning learns patterns from data.",
        "RAG retrieves relevant information.",
    ]

    embeddings = embed_texts(texts, batch_size=3)

    assert len(embeddings) == len(texts)

    for embedding in embeddings:
        assert len(embedding) == 384


def test_empty_texts():
    embeddings = embed_texts([])

    assert embeddings == []