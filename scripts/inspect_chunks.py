import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_all_documents
from src.chunker import chunk_documents


DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def main():

    print("=" * 70)
    print("DOCUMENT CHUNKING INSPECTION")
    print("=" * 70)
    print()

    documents = load_all_documents(DOCUMENTS_DIR)

    print()
    print(f"Documents loaded: {len(documents)}")
    print()

    chunks = chunk_documents(documents)

    print("=" * 70)
    print(f"TOTAL CHUNKS: {len(chunks):,}")
    print("=" * 70)

    print()

    for chunk in chunks[:10]:

        print("-" * 70)

        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk index: {chunk['chunk_index']}")

        print()
        print("CONTENT:")
        print(chunk["content"][:1000])

        print()


if __name__ == "__main__":
    main()