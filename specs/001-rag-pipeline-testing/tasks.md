---

description: "Task list for RAG Retrieval & Pipeline Testing"
---

# Tasks: RAG Retrieval & Pipeline Testing

**Input**: Design documents from `specs/001-rag-pipeline-testing/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests**: This task list includes test tasks as per best practices for ensuring quality.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the basic structure and entry points for the retrieval feature.

- [x] T001 Create retrieval module directory src/retrieval/
- [x] T002 [P] Create retriever.py skeleton file in src/retrieval/retriever.py
- [x] T003 [P] Create validation.py skeleton file in src/retrieval/validation.py
- [x] T004 Extend src/cli.py to include a 'retrieve' command argument parser
- [x] T005 [P] Create tests/unit/retrieval/ directory for unit tests
- [x] T006 [P] Create tests/integration/retrieval/ directory for integration tests

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Ensure core dependencies and configurations are accessible.

**âš ï¸ CRITICAL**: No user story work can begin until this phase is complete

- [x] T007 Ensure `ingestion_pipeline.config.Config` is properly imported and configured in `src/retrieval/retriever.py`
- [x] T008 Implement Cohere client initialization within `src/retrieval/retriever.py` (reusing patterns from ingestion_pipeline/embedder.py)
- [x] T009 Implement Qdrant client initialization within `src/retrieval/retriever.py` (reusing patterns from ingestion_pipeline/vector_store.py)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Validate RAG Retrieval with Query (Priority: P1) ðŸŽ¯ MVP

**Goal**: Enable executing queries against Qdrant to retrieve relevant chunks using Cohere embeddings.

**Independent Test**: Provide a query via CLI, verify top-k chunks, metadata, and logging.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T010 [P] [US1] Create unit test for Cohere query embedding generation in tests/unit/retrieval/test_retriever.py
- [x] T011 [P] [US1] Create unit test for Qdrant connection and basic search functionality in tests/unit/retrieval/test_retriever.py
- [x] T012 [P] [US1] Create integration test for `src/cli.py retrieve` command with a valid query in tests/integration/retrieval/test_cli_retrieval.py
- [x] T013 [P] [US1] Create integration test for `src/cli.py retrieve` command with an empty query in tests/integration/retrieval/test_cli_retrieval.py
- [x] T014 [P] [US1] Create integration test for `src/cli.py retrieve` command with a low-match query in tests/integration/retrieval/test_cli_retrieval.py

### Implementation for User Story 1

- [x] T015 [US1] Implement `CohereEmbedder` class for query embedding generation in `src/retrieval/retriever.py`
- [x] T016 [US1] Implement `QdrantRetriever` class with `retrieve_chunks` method in `src/retrieval/retriever.py`
- [x] T017 [US1] Modify `QdrantRetriever.retrieve_chunks` to use `CohereEmbedder` for query embedding in `src/retrieval/retriever.py`
- [x] T018 [US1] Implement top-k vector search logic within `QdrantRetriever.retrieve_chunks` in `src/retrieval/retriever.py`
- [x] T019 [US1] Extract and format retrieved chunk data, including all metadata (URL, title, section, text, chunk_index), in `src/retrieval/retriever.py`
- [x] T020 [US1] Add logic to `QdrantRetriever.retrieve_chunks` for graceful handling of empty or low-match queries in `src/retrieval/retriever.py`
- [x] T021 [US1] Implement logging of similarity scores and retrieval steps within `QdrantRetriever.retrieve_chunks` in `src/retrieval/retriever.py`
- [x] T022 [US1] Integrate `QdrantRetriever` and its `retrieve_chunks` method into the `retrieve` command in `src/cli.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T023 Review and refine `specs/001-rag-pipeline-testing/quickstart.md`
- [x] T024 Add comprehensive docstrings and type hints to `src/retrieval/retriever.py` and `src/retrieval/validation.py`
- [x] T025 Code cleanup and refactoring across the new retrieval module and `cli.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS User Story 1
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion
- **Polish (Final Phase)**: Depends on User Story 1 being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories

### Within Each User Story

- Tests MUST be written and FAIL before implementation (T010-T014 before T015-T022)
- `CohereEmbedder` implementation before `QdrantRetriever` uses it (T015 before T017)
- Core `QdrantRetriever` logic before integration into CLI (T016-T021 before T022)

### Parallel Opportunities

- All Setup tasks T002, T003, T005, T006 can run in parallel.
- All Foundational tasks T007, T008, T009 can run in parallel.
- Tests for User Story 1 (T010-T014) can run in parallel.
- Within User Story 1 implementation, tasks for different files can be parallelized, e.g., T015 (CohereEmbedder) and T016 (QdrantRetriever skeleton) could begin in parallel if their internal dependencies are clear.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational â†’ Foundation ready
2. Add User Story 1 â†’ Test independently â†’ Deploy/Demo (MVP!)

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
