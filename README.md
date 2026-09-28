```markdown
# AI RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) system built with Python, ChromaDB, Sentence Transformers, and Google Gemini.

The system loads a real document corpus, splits documents into overlapping token-based chunks, converts the chunks into vector embeddings, stores them in ChromaDB, retrieves semantically relevant chunks for a user query, and passes the retrieved context to Gemini to generate a grounded answer.

---

## Overview

Large Language Models can generate useful answers but may not have access to information contained in a specific private or custom document collection.

This project addresses that problem using Retrieval-Augmented Generation.

Instead of asking the language model to answer directly, the system:

1. Loads documents from the local corpus.
2. Splits the documents into overlapping chunks.
3. Generates vector embeddings for every chunk.
4. Stores the embeddings in ChromaDB.
5. Converts the user's question into an embedding.
6. Retrieves the most semantically relevant document chunks.
7. Builds a context from the retrieved chunks.
8. Sends the question and retrieved context to Gemini.
9. Generates an answer grounded in the retrieved information.
10. Displays the sources used for the answer.

---

## Architecture

```text
                    User Question
                          |
                          v
                Sentence Transformer
                   Query Embedding
                          |
                          v
                    ChromaDB
                 Semantic Retrieval
                          |
                          v
                 Top-K Document Chunks
                          |
                          v
                  Context Construction
                          |
                          v
              Google Gemini Generation
                          |
                          v
                  Grounded Answer
                          |
                          v
                   Source Attribution

```

## Project Structure

```text
ai-rag-document-chatbot/
│
├── documents/
│   ├── agentic_ai/
│   ├── python_documentation/
│   ├── deep_learning_documentation.md
│   ├── generative_ai_documentation.md
│   ├── machine_learning_documentation.md
│   └── rag_documentation.md
│
├── data/
│   └── chroma/
│       └── Generated locally during ingestion
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   └── rag_pipeline.py
│
├── scripts/
│   ├── ask.py
│   ├── collect_sources.py
│   ├── compare_rag.py
│   ├── evaluate_retrieval.py
│   ├── ingest.py
│   ├── inspect_chunks.py
│   ├── inspect_documents.py
│   ├── test_generator.py
│   ├── test_rag.py
│   └── test_retrieval.py
│
├── tests/
│   ├── __init__.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_generator.py
│   ├── test_retrieval.py
│   └── test_vector_store.py
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

## Document Corpus

The project uses a real technical-document corpus containing:

* Python documentation
* Machine learning documentation
* Deep learning documentation
* Generative AI documentation
* Retrieval-Augmented Generation documentation
* Agentic AI documentation

Current corpus statistics:

* Documents: 543
* Corpus size: approximately 15.53 MB

The corpus is intentionally heterogeneous so that semantic retrieval can be evaluated across different technical topics.

## Chunking

Documents are split into overlapping token-based chunks.

Current configuration:

* Chunk size: 220 tokens
* Chunk overlap: 40 tokens
* Tokenizer: `sentence-transformers/all-MiniLM-L6-v2`

The chunking strategy creates overlapping windows so that important information near chunk boundaries is less likely to be lost.

Each chunk receives metadata including:

* Source document
* Chunk index
* Chunk ID

Example:
`rag_documentation.md::chunk_5`

## Embeddings

The project uses:

* Model: `all-MiniLM-L6-v2`
* Embedding dimension: 384

Embeddings are normalized before being stored in ChromaDB. On the development machine, embedding generation uses CUDA when available.

Example development environment:

* GPU: NVIDIA GeForce RTX 4050 Laptop GPU
* CUDA: 12.8

The implementation automatically falls back to CPU when CUDA is unavailable.

## Vector Database

ChromaDB is used as the persistent vector database.

Collection:
`rag_documents`

The vector database stores:

* Chunk IDs
* Chunk text
* Chunk metadata
* Vector embeddings

The generated ChromaDB data is stored locally in:
`data/chroma/`

This directory is intentionally excluded from Git because it can be regenerated from the document corpus.

## Retrieval

For every question:

1. The query is converted into a 384-dimensional embedding.
2. ChromaDB performs semantic similarity search.
3. The top 5 results are retrieved.
4. Each result includes its source, chunk index, content, and distance.
5. The retrieved chunks are combined into the LLM context.

Example retrieval:

**Query:**
`What is Retrieval-Augmented Generation?`

**Top result:**

* Source: `rag_documentation.md`
* Chunk: `5`

## Generation

The retrieved document chunks are passed to Google Gemini.

The generation prompt instructs the model to:

* Use only the retrieved context as the factual source.
* Avoid inventing information.
* Avoid using outside knowledge to fill missing information.
* State when the provided documents do not contain enough information.

This helps keep the generated response grounded in the retrieved document collection.

## Source Attribution

The system returns the sources used to construct the answer.

Example:

```text
SOURCES

- rag_documentation.md
  chunk 5
  distance: 0.6486

- rag_documentation.md
  chunk 0
  distance: 0.7244

```

This makes the retrieval process inspectable rather than treating the generated answer as a black box.

## End-to-End RAG Demonstration

The complete pipeline was tested using:
`What is Retrieval-Augmented Generation and how does it work?`

The system successfully:

1. Embedded the query
2. Retrieved relevant chunks from ChromaDB
3. Built the context
4. Sent the context to Gemini
5. Generated a grounded answer
6. Reported the retrieved sources

The strongest retrieved result came from:

* Source: `rag_documentation.md`
* Chunk: `5`
* Distance: `0.6486`

The retrieved content describes RAG as combining a pretrained language model with an external data source through a pretrained neural retriever and explains that retrieved passages are used during inference.

## Without RAG vs With RAG

The project includes:
`scripts/compare_rag.py`

The experiment compares:

**Without RAG**
The question is sent directly to Gemini without providing the document collection.

**With RAG**
The question is processed through the retrieval pipeline:

```text
Question
   |
   v
Query Embedding
   |
   v
ChromaDB
   |
   v
Retrieved Document Chunks
   |
   v
Context Construction
   |
   v
Gemini
   |
   v
Grounded Answer

```

A completed retrieval experiment successfully retrieved five relevant chunks from `rag_documentation.md`.

The retrieved chunks included documentation describing:

* The pretrained language model
* External/non-parametric data
* Neural retrieval
* Retrieved passages
* Conditioning generation on retrieved passages

The RAG generation step successfully produced an answer grounded in the retrieved documentation.

**Experiment limitation**
The corresponding non-RAG generation depends on Gemini API availability and quota. During one recorded comparison run, the non-RAG request encountered a temporary Gemini API 503 UNAVAILABLE response after retries, while the retrieval and RAG generation steps completed successfully. Therefore, this experiment does not claim that the base LLM is unable to answer the question. Instead, it demonstrates the retrieval and grounding behavior of the RAG pipeline.

## Interactive Chatbot

After the vector database has been created, run:

```bash
python scripts/ask.py

```

Example:

```text
================================================================================
AI RAG DOCUMENT CHATBOT
================================================================================

Ask questions about the indexed document collection.
Type 'exit' or 'quit' to stop.

You: What is Retrieval-Augmented Generation?

Retrieving relevant documents...

Assistant:

Retrieval-Augmented Generation combines...

--------------------------------------------------------------------------------
SOURCES
--------------------------------------------------------------------------------
- rag_documentation.md (chunk 5, distance ...)
- rag_documentation.md (chunk 0, distance ...)

```

## Installation

1. **Clone the repository**

```bash
git clone [https://github.com/Vedant-1724/ai-rag-document-chatbot.git](https://github.com/Vedant-1724/ai-rag-document-chatbot.git)
cd ai-rag-document-chatbot

```

2. **Create a virtual environment**

Windows:

```bash
python -m venv venv
venv\Scripts\activate

```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate

```

3. **Install dependencies**

```bash
pip install -r requirements.txt

```

4. **Configure the Gemini API key**
Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

Never commit `.env` to Git.

## Build the Vector Database

Run:

```bash
python scripts/ingest.py

```

The ingestion pipeline:

```text
Documents
    |
    v
Document Loader
    |
    v
Token-Based Chunking
    |
    v
Sentence Transformer
    |
    v
Embeddings
    |
    v
ChromaDB

```

The vector database is generated locally under:
`data/chroma/`

The ChromaDB directory is excluded from Git because it can be regenerated from the document corpus.

## Run the Chatbot

After ingestion:

```bash
python scripts/ask.py

```

## Run the End-to-End RAG Test

```bash
python scripts/test_rag.py

```

This test displays:

* The user query
* Retrieved document chunks
* Source documents
* Chunk indexes
* Similarity distances
* Generated answer

The retrieval results are displayed before generation so that the retrieval stage can be independently inspected.

## Run Retrieval Evaluation

```bash
python scripts/evaluate_retrieval.py

```

This script evaluates retrieval behavior using multiple technical queries and displays the retrieved sources and similarity distances.

## Run the RAG Comparison Experiment

```bash
python scripts/compare_rag.py

```

This experiment compares direct LLM generation with generation using retrieved document context. It displays:

* The question
* Non-RAG generation result
* Retrieved chunks
* RAG generation result
* Retrieved sources

## Automated Tests

The project includes automated tests for:

* Document chunking
* Embedding generation
* Generator behavior
* Retrieval
* Vector store operations

Run:

```bash
pytest -v

```

Current test result:
`16 passed`

The generator unit tests use a mocked Gemini client so that automated tests do not consume Gemini API quota.

## Error Handling

The Gemini generator includes bounded retry handling for temporary server errors. Temporary server errors such as HTTP 503 are retried using exponential backoff. Quota and rate-limit errors such as HTTP 429 are handled separately instead of repeatedly retrying an exhausted quota.

## Retrieval Considerations

The document corpus contains information from multiple technical domains. Because of this, broad or ambiguous questions can occasionally retrieve semantically related but less relevant chunks. For example, a query may retrieve highly relevant RAG documentation along with a technically adjacent document.

Potential improvements include:

* Metadata filtering
* Hybrid keyword and vector search
* Cross-encoder reranking
* Query expansion
* Better corpus segmentation
* Domain-aware retrieval

## Reproducibility

The repository includes the document corpus used by the project. The generated ChromaDB index is intentionally not committed because it can be rebuilt locally.

To reproduce the vector database:

```bash
python scripts/ingest.py

```

Then run:

```bash
python scripts/ask.py

```

## Technologies

| Technology | Purpose |
| --- | --- |
| Python | Application language |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| Google Gemini | LLM generation |
| PyPDF | PDF document loading |
| Transformers | Tokenization |
| pytest | Automated testing |
| python-dotenv | Environment configuration |

## Future Improvements

Potential future improvements include:

* Hybrid retrieval
* Cross-encoder reranking
* Conversation memory
* Streaming generation
* Web interface
* Document upload functionality
* Metadata-aware filtering
* Retrieval evaluation metrics
* Multi-query retrieval
* Improved citation formatting
* Production vector database deployment

## License

This project is licensed under the MIT License. See LICENSE.

## Author

Vedant Joshi
GitHub: https://github.com/Vedant-1724

```

```