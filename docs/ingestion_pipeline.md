# RAG Ingestion Pipeline Documentation

This document provides detailed information about the RAG Ingestion Pipeline.

## Overview

The RAG Ingestion Pipeline is designed to process content from Docusaurus sites, extract relevant information, generate embeddings, and store them in a Qdrant vector database.

## Components

- **Configuration**: Handles environment variable loading and validation.
- **Crawler**: Discovers and filters internal Docusaurus documentation URLs.
- **Extractor**: Fetches HTML content, parses it, extracts main content, and performs text normalization.
- **Chunker**: Splits extracted text into semantic chunks with configurable size and overlap.
- **Embedder**: Generates Cohere embeddings for text chunks, with batching and error handling.
- **Vector Store**: Manages interaction with the Qdrant vector database, including collection creation and upsert operations.
- **CLI**: Provides a command-line interface for initiating ingestion and performing validation tasks.

## Usage

Refer to the `quickstart.md` for a guide on setting up and running the pipeline.

## Further Details

[Add more detailed sections as needed for configuration, architecture, etc.]
