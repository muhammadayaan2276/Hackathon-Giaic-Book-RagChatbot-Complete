# Research: RAG Agent Implementation

**Feature**: `005-rag-agent-qdrant`

This document summarizes the technical decisions made to resolve ambiguities in the implementation plan.

## 1. Qdrant Connection Details

-   **Decision**: The agent will connect to Qdrant using a URL and an API key. These credentials must be provided as environment variables: `QDRANT_URL` and `QDRANT_API_KEY`.
-   **Rationale**: This approach avoids hardcoding secrets, which is a security best practice. It is the standard method used throughout this project, as seen in `src/ingestion_pipeline/config.py` and `src/retrieval/retriever.py`.
-   **Alternatives Considered**: Hardcoding credentials (rejected for security reasons), using a configuration file (rejected as environment variables are simpler and standard for this project).

## 2. Qdrant Collection Name

-   **Decision**: The agent will query the collection named `docusaurus_book`.
-   **Rationale**: The existing `src/retrieval/retriever.py` already contains a hardcoded default for this collection name, establishing a clear precedent. Using a consistent name is essential for connecting the retrieval component with the ingestion pipeline.
-   **Alternatives Considered**: Making the collection name a command-line argument. This was rejected for the initial implementation to keep the CLI simple, but could be added later if needed.

## 3. Embedding Model

-   **Decision**: The agent will use Cohere's `embed-english-v3.0` model with an input type of `search_query` to generate embeddings for user queries.
-   **Rationale**: This is the exact model used in the existing `CohereEmbedder` class in `src/retrieval/retriever.py`. Using the same embedding model for querying as was used for ingestion is a strict requirement for vector search to work correctly.
-   **Alternatives Considered**: None. Using a different model would result in incorrect and meaningless search results.
