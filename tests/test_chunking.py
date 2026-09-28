from src.chunker import chunk_text, chunk_documents


def test_chunk_text_returns_multiple_chunks():
    text = "Python is a programming language. " * 200

    chunks = chunk_text(
        text,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_invalid_overlap():
    text = "Python is useful."

    try:
        chunk_text(
            text,
            chunk_size=20,
            chunk_overlap=20,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_chunk_documents_preserves_metadata():
    documents = [
        {
            "source": "test.md",
            "content": "Machine learning is useful. " * 100,
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk["source"] == "test.md"
        assert "chunk_index" in chunk
        assert "chunk_id" in chunk
        assert "content" in chunk