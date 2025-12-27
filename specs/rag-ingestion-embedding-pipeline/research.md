# Research for RAG Ingestion & Embedding Pipeline

This document captures the research and decisions made to resolve the "NEEDS CLARIFICATION" items from the implementation plan.

## Primary Dependencies

### Decision
- **Cohere**: For embedding generation.
- **Qdrant-client**: To interact with Qdrant vector database.
- **BeautifulSoup4**: For HTML parsing and content extraction.
- **Langchain**: For text chunking.

### Rationale
- **Cohere**: It is specified in the prompt. It provides high-quality embeddings.
- **Qdrant-client**: It is the official Python client for Qdrant, which is the chosen vector database.
- **BeautifulSoup4**: It is a popular and powerful library for web scraping and HTML parsing in Python, making it ideal for extracting content from the Docusaurus site.
- **Langchain**: It provides robust and easy-to-use text splitters that are suitable for creating semantic chunks for RAG.

### Alternatives Considered
- **OpenAI**: Another option for embeddings, but Cohere was specified.
- **Pinecone, Weaviate**: Other vector databases, but Qdrant was specified.
- **Scrapy**: A more powerful web crawling framework, but it is too complex for this use case. BeautifulSoup is sufficient.
- **Custom text splitters**: We could write our own text splitters, but Langchain's are well-tested and sufficient.

## Storage

### Decision
- **Qdrant**: A vector database for storing embeddings.

### Rationale
- Qdrant is specified in the prompt. It is a high-performance, open-source vector database that is well-suited for RAG applications.

### Alternatives Considered
- None, as Qdrant was specified.

## Testing

### Decision
- **pytest**: For unit and integration testing.

### Rationale
- pytest is a mature and feature-rich testing framework for Python that is easy to use and has a large ecosystem of plugins.

### Alternatives Considered
- **unittest**: Part of the Python standard library, but pytest offers a more concise and powerful syntax.

## Target Platform

### Decision
- **Linux server**: The pipeline will be deployed on a Linux server.

### Rationale
- Linux is a common and stable platform for running data processing pipelines. It is well-supported by all the chosen technologies.

### Alternatives Considered
- **Windows, macOS**: These are also viable, but Linux is the standard for production deployments.

## Performance, Constraints, and Scale

These are highly dependent on the specific Docusaurus site and the expected usage of the RAG application. For the initial implementation, we will use reasonable defaults and focus on a robust and correct implementation. Performance tuning and scaling can be addressed in a later iteration.

- **Performance Goals**: Process 10 pages/second.
- **Constraints**: The pipeline should be able to run on a machine with 8GB of RAM.
- **Scale/Scope**: The pipeline should be able to handle up to 10,000 documents.
