# Quickstart Guide for RAG Ingestion Pipeline

This guide explains how to set up and run the RAG ingestion pipeline.

## Prerequisites

- Python 3.11
- An active internet connection
- Access to a Cohere API key
- A running instance of Qdrant

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables**:
    Create a `.env` file in the root of the project and add the following:
    ```
    COHERE_API_KEY="<your_cohere_api_key>"
    QDRANT_HOST="<your_qdrant_host>"
    QDRANT_PORT="<your_qdrant_port>"
    ```

## Running the Pipeline

The ingestion pipeline is run from the command line.

```bash
python -m src.cli.ingest --url <docusaurus_site_url>
```

### Arguments

-   `--url`: (Required) The base URL of the Docusaurus site to crawl.

### Example

```bash
python -m src.cli.ingest --url https://my-docusaurus-book.com
```

## Validation

After the pipeline has run, you can verify that the data has been ingested by querying the Qdrant database. You can also run the sample similarity queries provided in the validation script.

```bash
python -m src.cli.validate --query "your search query"
```
