from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (RAG-Document-Collector/1.0)"
}


SOURCES = {
    "machine_learning_documentation.md": {
        "title": "Machine Learning Documentation",
        "sources": [
            (
                "Scikit-learn User Guide",
                "https://scikit-learn.org/stable/user_guide.html",
            ),
            (
                "Supervised Learning",
                "https://scikit-learn.org/stable/supervised_learning.html",
            ),
            (
                "Unsupervised Learning",
                "https://scikit-learn.org/stable/unsupervised_learning.html",
            ),
            (
                "Preprocessing Data",
                "https://scikit-learn.org/stable/modules/preprocessing.html",
            ),
            (
                "Model Selection and Evaluation",
                "https://scikit-learn.org/stable/model_selection.html",
            ),
        ],
    },

    "deep_learning_documentation.md": {
        "title": "Deep Learning Documentation",
        "sources": [
            (
                "PyTorch Learn the Basics",
                "https://docs.pytorch.org/tutorials/beginner/basics/intro.html",
            ),
            (
                "PyTorch Quickstart",
                "https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html",
            ),
            (
                "PyTorch Tensors",
                "https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
            ),
            (
                "Datasets and DataLoaders",
                "https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html",
            ),
            (
                "Building the Neural Network",
                "https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
            ),
            (
                "Automatic Differentiation",
                "https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html",
            ),
            (
                "Optimization",
                "https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
            ),
            (
                "Save and Load the Model",
                "https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html",
            ),
        ],
    },

    "generative_ai_documentation.md": {
        "title": "Generative AI Documentation",
        "sources": [
            (
                "Gemini API Documentation",
                "https://ai.google.dev/gemini-api/docs",
            ),
            (
                "Getting Started",
                "https://ai.google.dev/gemini-api/docs/get-started",
            ),
            (
                "Embeddings",
                "https://ai.google.dev/gemini-api/docs/embeddings",
            ),
            (
                "Document Understanding",
                "https://ai.google.dev/gemini-api/docs/document-processing",
            ),
            (
                "Function Calling",
                "https://ai.google.dev/gemini-api/docs/function-calling",
            ),
            (
                "Structured Outputs",
                "https://ai.google.dev/gemini-api/docs/structured-output",
            ),
        ],
    },

    "rag_documentation.md": {
        "title": "Retrieval-Augmented Generation Documentation",
        "sources": [
            (
                "Hugging Face RAG",
                "https://huggingface.co/docs/transformers/main/model_doc/rag",
            ),
            (
                "Hugging Face RAG with Chat Templates, Tools and Documents",
                "https://huggingface.co/docs/transformers/main/chat_template_tools_and_documents",
            ),
            (
                "Hugging Face Advanced Chat Templates",
                "https://huggingface.co/docs/transformers/main/chat_template_advanced",
            ),
            (
                "Hugging Face Tokenizer Documentation",
                "https://huggingface.co/docs/transformers/main_classes/tokenizer",
            ),
        ],
    },
}


def clean_text(html: str) -> str:
    """Extract readable text from an HTML documentation page."""

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that don't contain useful documentation text.
    for element in soup([
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "svg",
    ]):
        element.decompose()

    # Prefer the main documentation content.
    main = soup.find("main")

    if main:
        text = main.get_text("\n", strip=True)
    else:
        article = soup.find("article")

        if article:
            text = article.get_text("\n", strip=True)
        else:
            text = soup.get_text("\n", strip=True)

    # Normalize excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove excessive spaces.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def fetch_page(title: str, url: str) -> str:
    """Download and extract text from one documentation page."""

    print(f"Downloading: {title}")
    print(f"URL: {url}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    text = clean_text(response.text)

    if len(text) < 200:
        raise ValueError(
            f"Very little text extracted from {url}. "
            "The page structure may have changed."
        )

    return text


def build_document(filename: str, config: dict) -> None:
    """Build one Markdown document from multiple official sources."""

    output_path = DOCUMENTS_DIR / filename

    sections = []

    sections.append(f"# {config['title']}\n")

    sections.append(
        "> This document was collected from official documentation "
        "sources for use as a Retrieval-Augmented Generation (RAG) corpus.\n"
    )

    sections.append(
        "## Sources\n\n"
    )

    for title, url in config["sources"]:
        sections.append(f"- [{title}]({url})")

    sections.append("\n\n---\n")

    successful = 0

    for title, url in config["sources"]:
        try:
            text = fetch_page(title, url)

            sections.append(f"\n## {title}\n")
            sections.append(f"**Source:** {url}\n")
            sections.append(text)
            sections.append("\n\n---\n")

            successful += 1

        except Exception as error:
            print(f"WARNING: Could not collect {url}")
            print(f"Reason: {error}")

    output_path.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )

    word_count = len(
        re.findall(
            r"\b[\w'-]+\b",
            output_path.read_text(encoding="utf-8"),
        )
    )

    print()
    print(f"Created: {output_path}")
    print(f"Successful sources: {successful}/{len(config['sources'])}")
    print(f"Word count: {word_count}")
    print("-" * 60)


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OFFICIAL DOCUMENTATION COLLECTOR")
    print("=" * 60)
    print()

    for filename, config in SOURCES.items():
        build_document(filename, config)

    print()
    print("=" * 60)
    print("SOURCE COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()