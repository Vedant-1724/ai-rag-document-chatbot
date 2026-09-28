import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import generate_answer


def main():
    context = """
    Retrieval-Augmented Generation combines a pretrained
    language model with an external data source.
    The system retrieves relevant passages and conditions
    generation on those passages during inference.
    """

    question = (
        "What is Retrieval-Augmented Generation "
        "and how does it work?"
    )

    print("=" * 80)
    print("GEMINI GENERATION TEST")
    print("=" * 80)
    print()

    print(f"Question: {question}")
    print()
    print("Generating answer...")
    print()

    answer = generate_answer(
        question=question,
        context=context,
    )

    print("=" * 80)
    print("GENERATED ANSWER")
    print("=" * 80)
    print()
    print(answer)
    print()


if __name__ == "__main__":
    main()