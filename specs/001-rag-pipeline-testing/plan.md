# Implementation Plan: RAG Retrieval & Pipeline Testing

**Branch**: `001-rag-pipeline-testing` | **Date**: 2025-12-27 | **Spec**: specs/001-rag-pipeline-testing/spec.md
**Input**: Feature specification from `/specs/001-rag-pipeline-testing/spec.md`

## Summary

Retrieval pipeline to validate embedded Docusaurus book content in Qdrant using Cohere query embeddings, ensuring correct retrieval of top-k relevant chunks and metadata for integration with Spec 3 agents. The plan outlines the development of a working retrieval test script capable of loading environment variables, initializing clients, generating query embeddings, performing vector search, validating results, and logging outcomes, all while handling various query scenarios gracefully.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: `qdrant-client`, `cohere`, `langchain-text-splitters`, `requests`, `python-dotenv`
**Storage**: Qdrant Cloud
**Testing**: `pytest`
**Target Platform**: Linux server
**Project Type**: Single project (Python script/CLI)
**Performance Goals**: Responsive retrieval, with a focus on functional correctness and validation within the given timeline.
**Constraints**: Python, Cohere Embeddings, Qdrant Cloud, Deterministic Retrieval, No LLM Generation, Timeline: complete within 1-2 tasks.
**Scale/Scope**: Validation of RAG retrieval for Docusaurus book content, focusing on accuracy and metadata for integration with Spec 3 agents.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project's constitution template (`.specify/memory/constitution.md`) needs to be finalized before a comprehensive Constitution Check can be performed. Assuming the general principles of software engineering (e.g., testability, modularity) are adopted.

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-pipeline-testing/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
src/
├── cli/                 # Existing CLI for ingestion, will add retrieval command
├── ingestion_pipeline/  # Existing ingestion pipeline
├── retrieval/           # New directory for retrieval logic
│   └── retriever.py     # Main retrieval logic
│   └── validation.py    # For validating results
└── services/            # Existing services

tests/
├── unit/                # Unit tests for retrieval logic
└── integration/         # Integration tests for end-to-end retrieval
```

**Structure Decision**: Option 1: Single project. A new `retrieval` directory will be created under `src/` to house retrieval-specific logic (e.g., `retriever.py` and `validation.py`). Existing `src/cli/` will be extended to include commands for this retrieval pipeline testing.

## Phase 0: Outline & Research

### Research

Based on the detailed feature specification, the technical context, and the well-defined constraints (Python, Cohere, Qdrant Cloud), no further specific research is identified as immediately necessary for the initial implementation of the retrieval test script. The core technologies are already established and utilized within the project's ingestion pipeline.

## Phase 1: Design & Contracts

### Data Model

The data model for the retrieval pipeline primarily revolves around two key entities: `Query` and `Chunk`.

-   **Query**:
    *   Represents the input text provided by the user for retrieval.
    *   Attributes: `text` (string).
    *   No complex validation beyond ensuring it's a non-empty string for practical queries.

-   **Chunk**:
    *   Represents a segment of text extracted from the Docusaurus book, stored in Qdrant along with its embedding and metadata.
    *   Attributes (aligned with `spec.md` and Qdrant payload):
        *   `id` (string/UUID): Unique identifier for the chunk, corresponding to the Qdrant point ID.
        *   `text` (string): The actual content of the chunk.
        *   `embedding` (list of floats): Vector representation of the chunk text. (Handled by Cohere, not directly stored in the payload).
        *   `url` (string): Source URL of the Docusaurus page.
        *   `title` (string): Title of the source document/page.
        *   `section` (string): Specific section within the document (if derivable).
        *   `chunk_index` (integer): Order of the chunk within its source document.
    *   Relationships: Each `Chunk` is associated with a `Query` during retrieval (via similarity), but they are independently stored entities.

### Contracts

For this internal retrieval test script, formal API contracts (e.g., OpenAPI schemas) are not required. The interaction will be direct Python function calls.

### Quickstart

A `quickstart.md` will be created to provide instructions on how to set up the environment and run the retrieval test script. This will include details on:
1.  Ensuring environment variables (COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY) are set.
2.  How to execute the retrieval command via the `cli.py` (e.g., `python src/cli.py retrieve --query "your question" --top-k 5`).
3.  Expected output format.

## Agent Context Update

After creating the necessary files for Phase 1, the agent context will be updated to include any new technology from this plan that is relevant for future agent interactions. Specifically, Cohere, Qdrant, and `langchain-text-splitters` are already known, but emphasizing their role in retrieval could be beneficial.
