import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_all_documents
from src.chunker import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    get_chroma_client,
    get_collection_count,
    add_chunks,
)


DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# Number of chunks embedded at one time.
EMBEDDING_BATCH_SIZE = 32


def reset_vector_store():
    """Delete the existing ChromaDB collection and recreate it."""

    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        print("No existing collection found. Creating a new one...")

    # Recreate the collection through the existing helper.
    from src import vector_store

    vector_store._collection = None

    vector_store.get_collection()

    print("Fresh ChromaDB collection ready.")


def ingest():
    print("=" * 70)
    print("RAG DOCUMENT INGESTION")
    print("=" * 70)
    print()

    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"ChromaDB directory: {CHROMA_DIR}")
    print()

    # ---------------------------------------------------------------
    # STEP 1 — Reset test data
    # ---------------------------------------------------------------

    print("[1/4] Resetting ChromaDB...")
    reset_vector_store()
    print()

    # ---------------------------------------------------------------
    # STEP 2 — Load documents
    # ---------------------------------------------------------------

    print("[2/4] Loading documents...")
    documents = load_all_documents(DOCUMENTS_DIR)

    print()
    print(f"Documents loaded: {len(documents):,}")
    print()

    # ---------------------------------------------------------------
    # STEP 3 — Create chunks
    # ---------------------------------------------------------------

    print("[3/4] Creating chunks...")
    chunks = chunk_documents(documents)

    print()
    print(f"Total chunks created: {len(chunks):,}")
    print()

    # ---------------------------------------------------------------
    # STEP 4 — Embed and store
    # ---------------------------------------------------------------

    print("[4/4] Embedding and storing chunks...")
    print()
    print(f"Embedding batch size: {EMBEDDING_BATCH_SIZE}")
    print()

    total_chunks = len(chunks)

    for start in range(0, total_chunks, EMBEDDING_BATCH_SIZE):

        end = min(
            start + EMBEDDING_BATCH_SIZE,
            total_chunks,
        )

        batch_chunks = chunks[start:end]

        texts = [
            chunk["content"]
            for chunk in batch_chunks
        ]

        print(
            f"Embedding chunks "
            f"{start + 1:,}-{end:,} "
            f"of {total_chunks:,}"
        )

        embeddings = embed_texts(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        add_chunks(
            batch_chunks,
            embeddings,
        )

        print(
            f"Stored: {get_collection_count():,} chunks"
        )
        print()

    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print()
    print(f"Documents: {len(documents):,}")
    print(f"Chunks:    {len(chunks):,}")
    print(f"Stored:    {get_collection_count():,}")
    print()
    print(f"ChromaDB:  {CHROMA_DIR}")
    print()


if __name__ == "__main__":
    ingest()