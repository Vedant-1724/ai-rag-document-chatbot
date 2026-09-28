import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_all_documents


DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def main():
    print("=" * 70)
    print("DOCUMENT LOADER INSPECTION")
    print("=" * 70)
    print()

    documents = load_all_documents(DOCUMENTS_DIR)

    print()
    print("=" * 70)
    print(f"TOTAL DOCUMENTS LOADED: {len(documents)}")
    print("=" * 70)

    total_characters = 0
    total_words = 0

    for document in documents:
        words = len(document["content"].split())

        total_characters += len(document["content"])
        total_words += words

        print()
        print(f"Source: {document['source']}")
        print(f"Type:   {document['file_type']}")
        print(f"Words:  {words:,}")

    print()
    print("=" * 70)
    print(f"TOTAL CHARACTERS: {total_characters:,}")
    print(f"TOTAL WORDS:      {total_words:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()