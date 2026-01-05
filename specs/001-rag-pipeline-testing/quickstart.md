# Quickstart: RAG Retrieval & Pipeline Testing

**Date**: 2025-12-27

This guide provides instructions on how to set up your environment and execute the RAG Retrieval Pipeline Testing script.

## 1. Prerequisites

Before running the retrieval script, ensure the following:

*   **Python 3.11+** is installed.
*   **Dependencies**: All Python dependencies are installed. You can install them by running:
    ```bash
    pip install -r requirements.txt
    ```
*   **Ingestion Pipeline Executed**: The Docusaurus book content must have been previously ingested into your Qdrant instance. Refer to the ingestion pipeline documentation for details.
*   **.env file**: A `.env` file must be present in the project root directory (`C:\Users\pc\Desktop\RagChatbot\hackathon-Giaic\`) with the following environment variables correctly configured:
    ```env
    COHERE_API_KEY="YOUR_COHERE_API_KEY"
    QDRANT_URL="YOUR_QDRANT_CLUSTER_URL"
    QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
    ```
    *   **`COHERE_API_KEY`**: Your API key for Cohere embeddings.
    *   **`QDRANT_URL`**: The full URL to your Qdrant Cloud instance (e.g., `https://<cluster-id>.us-east4-0.gcp.cloud.qdrant.io:6333`).
    *   **`QDRANT_API_KEY`**: Your API key for authenticating with Qdrant Cloud.

## 2. Running the Retrieval Test Script

The retrieval test script is integrated into the existing `cli.py`.

### Basic Retrieval

To perform a basic retrieval query, use the `retrieve` command:

```bash
python src/cli.py retrieve --query "What is Robotic AI?" --top-k 5
```

*   Replace `"What is Robotic AI?"` with your desired query string.
*   Adjust `--top-k` to specify the number of top relevant chunks you wish to retrieve (default is usually 5).

### Example Output

The script will log the retrieval process and output details of the retrieved chunks, including their text, URL, title, and similarity score.

```
# Example log output (simplified)
INFO - CLI: Running retrieve command for query: What is Robotic AI?
INFO - Generating embeddings for query...
INFO - Query embeddings generated.
INFO - Retrieving top 5 chunks from Qdrant...
INFO - Retrieved 5 chunks.
INFO - Chunk 1: Score=0.85, Title=Chapter 1: AI Decision Pipelines, URL=...
DEBUG - Content: "..."
INFO - Chunk 2: Score=0.82, Title=Module 3: AI Robot Brain, URL=...
DEBUG - Content: "..."
...
```

## 3. Interpreting Results

*   **Similarity Score**: A higher score indicates greater relevance to the query.
*   **Metadata**: Verify that the URL, title, and other metadata accurately reflect the source of the retrieved content.
*   **Content**: Review the `text` of the retrieved chunks to manually assess their relevance to your query.

This quickstart aims to provide a straightforward way to validate the RAG retrieval pipeline's functionality. For more advanced validation scenarios, refer to the detailed specification and implementation.
