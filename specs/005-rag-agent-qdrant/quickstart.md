# Quickstart: RAG Agent CLI

**Feature**: `005-rag-agent-qdrant`

This guide explains how to set up and run the RAG agent.

## Prerequisites

-   Python 3.9+
-   An active Python virtual environment.
-   The Qdrant database must be populated with book content from the ingestion pipeline.

## 1. Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```
*(Note: Ensure `openai`, `qdrant-client`, and `typer` are listed in `requirements.txt`)*

## 2. Configuration

Export the following environment variables. You can add them to a `.env` file and use `python-dotenv`.

```bash
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."
export QDRANT_URL="https://<your-cluster-id>.gcp.cloud.qdrant.io:6333"
export QDRANT_API_KEY="..."
```

## 3. Running the Agent

You can run the agent from the command line to ask a question.

```bash
python src/retrieval/agent.py "What is the main purpose of a ROS 2 node?"
```

### Example Output (Success)

```
INFO: Generating embeddings for query: 'What is the main purpose of a ROS 2 node?'
INFO: Performing top-5 vector search in Qdrant collection 'docusaurus_book'
INFO: Retrieved 5 chunks for query: 'What is the main purpose of a ROS 2 node?'
INFO: Generating answer...

The main purpose of a ROS 2 node is to be a fundamental building block of a ROS system. It is a process that performs computation, such as controlling a motor, reading sensor data, or planning a path. Nodes communicate with each other by sending messages via topics, services, or actions.

Sources:
1. [Score: 0.89] Chapter 1: ROS 2 Fundamentals (url:/module1/chapter1)
2. [Score: 0.85] Chapter 2: Python Agent Bridges (url:/module1/chapter2)
```

### Example Output (Not Found)

```
INFO: Generating embeddings for query: 'What is the best recipe for apple pie?'
INFO: Performing top-5 vector search in Qdrant collection 'docusaurus_book'
INFO: No relevant chunks found for the query.

I could not find any information about "What is the best recipe for apple pie?" in the provided book content. My knowledge is strictly limited to the technical book.
```
