# Tasks: Retrieval-Augmented AI Agent

**Input**: Design documents from `specs/005-rag-agent-qdrant/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Organization**: Tasks are grouped by phase. Since there is only one primary user story, it constitutes the main implementation phase.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1)
- Include exact file paths in descriptions.

---

## Phase 1: Setup

**Purpose**: Project initialization and basic file structure.

- [x] T001 Create the main agent file at `src/retrieval/agent.py` with placeholder content.
- [x] T002 Update `requirements.txt` to include `openai`, `qdrant-client`, `typer`, `cohere`, and `python-dotenv`.
- [x] T003 Create a `.env.example` file in the project root, listing `OPENAI_API_KEY`, `COHERE_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before the main logic can be implemented.

- [x] T004 Implement the basic CLI structure in `src/retrieval/agent.py` using `typer` to accept a query string as an argument.
- [x] T005 Implement a configuration loading mechanism in `src/retrieval/agent.py` to securely load environment variables from a `.env` file.
- [x] T006 Create the unit test file at `tests/unit/retrieval/test_agent_unit.py` with initial setup and imports.
- [x] T007 Create the integration test file at `tests/integration/retrieval/test_agent_integration.py` with initial setup and imports.

---

## Phase 3: User Story 1 - RAG Agent CLI (Priority: P1) 🎯 MVP

**Goal**: A user can ask a question via the CLI and receive a grounded answer from the book or a "not found" message.

**Independent Test**: Execute `python src/retrieval/agent.py "test query"` and verify the response is either an accurate, sourced answer or the correct fallback message, as detailed in `quickstart.md`.

### Implementation for User Story 1

- [x] T008 [P] [US1] Implement the `CohereEmbedder` class in `src/retrieval/agent.py` to generate embeddings for user queries.
- [x] T009 [P] [US1] Implement the `QdrantRetriever` class structure in `src/retrieval/agent.py`.
- [x] T010 [US1] Implement the Qdrant client connection logic within the `QdrantRetriever` class in `src/retrieval/agent.py`, using the loaded configuration.
- [x] T011 [US1] Implement the `retrieve_chunks` method in the `QdrantRetriever` class to perform a vector search and return relevant chunks from Qdrant, dependent on T008.
- [x] T012 [US1] Implement the main agent logic in `src/retrieval/agent.py` to initialize the OpenAI client. The system prompt must strictly instruct the agent to use only the provided context.
- [x] T013 [US1] Implement logic to format the retrieved chunks into a context string to be injected into the agent's prompt in `src/retrieval/agent.py`.
- [x] T014 [US1] Implement the response generation logic that calls the OpenAI agent with the formatted context and the original user query in `src/retrieval/agent.py`.
- [x] T015 [US1] Implement the fallback logic to return a "not found" message when no chunks are retrieved in `src/retrieval/agent.py`.
- [x] T016 [US1] Integrate all components (CLI, config, retriever, agent) into the main execution flow in the CLI function within `src/retrieval/agent.py`.

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect quality, reliability, and usability.

- [x] T017 [P] Add detailed INFO and ERROR level logging throughout the execution flow in src/retrieval/agent.py.
- [x] T018 [P] Add unit tests for the `CohereEmbedder` and `QdrantRetriever` classes in `tests/unit/retrieval/test_agent_unit.py`, mocking external API calls.
- [x] T019 Add an integration test in `tests/integration/retrieval/test_agent_integration.py` that mocks the Qdrant and OpenAI APIs to verify the end-to-end logic.
- [x] T020 Update the project's main `README.md` to include a section on how to set up and run the RAG agent CLI, referencing `quickstart.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion.
- **User Story 1 (Phase 3)**: Depends on Foundational completion.
- **Polish (Phase 4)**: Can be done in parallel with Phase 3, but is best after the core logic is stable.

### Within User Story 1

- `T008` (Embedder) and `T009` (Retriever class) can start in parallel.
- `T010` (Qdrant connection) depends on `T009`.
- `T011` (Retrieve chunks) depends on `T008` and `T010`.
- The main agent logic (`T012` to `T015`) can be developed in parallel with retrieval, but `T016` (integration) depends on all of them.

### Parallel Opportunities

- Once the Foundational phase is complete, developers can work on different parts of User Story 1. For example:
  - Developer A can work on `T008` and the unit tests in `T018`.
  - Developer B can work on `T009-T011` and the integration tests in `T019`.
  - Developer C can work on the core agent and prompting logic `T012-T015`.

---

## Implementation Strategy

### MVP First (User Story 1)

1.  Complete Phase 1 (Setup) and Phase 2 (Foundational).
2.  Complete all tasks in Phase 3 (User Story 1).
3.  **STOP and VALIDATE**: Manually run the CLI as described in `quickstart.md` to confirm the agent responds correctly to both in-scope and out-of-scope questions.
4.  Once validated, the core feature is complete. The tasks in Phase 4 can be addressed to improve robustness.
