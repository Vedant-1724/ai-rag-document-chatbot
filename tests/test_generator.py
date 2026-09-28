from src.generator import generate_answer


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def generate_content(self, model, contents):
        return FakeResponse(
            "RAG combines a language model with an external "
            "knowledge source and uses retrieved passages "
            "during generation."
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_generator_returns_answer(monkeypatch):

    context = """
    Retrieval-Augmented Generation combines a language model
    with an external knowledge source. The system retrieves
    relevant passages and provides them to the language model
    during generation.
    """

    question = "What is Retrieval-Augmented Generation?"

    monkeypatch.setattr(
        "src.generator.get_client",
        lambda: FakeClient(),
    )

    answer = generate_answer(
        question=question,
        context=context,
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "RAG" in answer


def test_generator_rejects_empty_question():

    context = "Some retrieved context."

    try:
        generate_answer(
            question="",
            context=context,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_generator_rejects_empty_context():

    question = "What is RAG?"

    try:
        generate_answer(
            question=question,
            context="",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass