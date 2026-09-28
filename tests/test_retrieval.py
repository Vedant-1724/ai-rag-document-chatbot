from src.retriever import retrieve


def test_retrieve_returns_results():
    results = retrieve(
        "What is Retrieval-Augmented Generation?",
        top_k=5,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 5


def test_retrieved_chunk_has_expected_fields():
    results = retrieve(
        "What is Retrieval-Augmented Generation?",
        top_k=3,
    )

    assert results

    result = results[0]

    assert "chunk_id" in result
    assert "content" in result
    assert "source" in result
    assert "chunk_index" in result
    assert "distance" in result


def test_retrieval_returns_rag_documentation():
    results = retrieve(
        "What is Retrieval-Augmented Generation and how does it work?",
        top_k=5,
    )

    assert results

    sources = [result["source"] for result in results]

    assert "rag_documentation.md" in sources


def test_retrieve_rejects_empty_query():
    try:
        retrieve("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_retrieve_rejects_invalid_top_k():
    try:
        retrieve(
            "What is RAG?",
            top_k=0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass