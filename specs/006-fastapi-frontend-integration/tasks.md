# Tasks: Backend-Frontend Integration for RAG Chatbot

This document outlines the tasks required to integrate the FastAPI backend with the Docusaurus frontend for the RAG chatbot.

## Dependencies

The user stories can be implemented in parallel, but both depend on the setup phase.

-   `US1` -> `Setup`
-   `US2` -> `Setup`

## Implementation Strategy

The implementation will follow an MVP-first approach. The initial focus will be on the backend setup and the full-book query user story (US1). The selected-text query user story (US2) can be implemented in parallel or after US1 is complete.

---

## Phase 1: Setup

This phase focuses on setting up the backend application.

- [x] T001 Create `api.py` file at the project root.
- [x] T002 Initialize a FastAPI app in `api.py`.
- [x] T003 Configure CORS middleware in `api.py` to allow requests from `http://localhost:3000`.

## Phase 2: User Story 1 - Full-Book Query

**Goal**: A developer can ask a general question and get a book-grounded answer.

- [x] T004 [US1] Define the request and response schemas for the chat API in `api.py`.
- [x] T005 [US1] Implement the `POST /chat` endpoint in `api.py`.
- [x] T006 [US1] Import the strict RAG agent from `strict_rag_agent.py` into `api.py`.
- [x] T007 [US1] Invoke the RAG agent within the `/chat` endpoint, passing the user's query.
- [x] T008 [P] [US1] Create a new React component `Chatbot` in `My-Book/src/components/Chatbot.js`.
- [x] T009 [US1] Add a basic UI to the `Chatbot` component with an input field, a submit button, and a message list area.
- [x] T010 [US1] Implement state management in the `Chatbot` component for the input, messages, and loading state.
- [x] T011 [US1] Connect the `Chatbot` component's submit action to the `/chat` endpoint using `fetch`.
- [x] T012 [US1] Render the assistant's response in the message list.
- [x] T013 [US1] Integrate the `Chatbot` component into the Docusaurus layout.

## Phase 3: User Story 2 - Selected-Text Query

**Goal**: A developer can ask a question about a selected piece of text and get a contextual answer.

- [x] T014 [US2] Update the `/chat` endpoint in `api.py` to handle the optional `selected_text` field.
- [x] T015 [US2] Update the RAG agent invocation to use `selected_text` as context if provided.
- [x] T016 [P] [US2] Implement a mechanism in the Docusaurus frontend to capture selected text.
- [x] T017 [US2] Create a UI element (e.g., a button) that appears when text is selected.
- [x] T018 [US2] Update the `Chatbot` component to accept the selected text and send it to the backend.

## Phase 4: Testing & Polish

- [x] T019 Test the full-book query flow end-to-end.

- [x] T020 Test the selected-text query flow end-to-end.

- [x] T021 Verify that responses are strictly from book content.

- [x] T022 Implement and test graceful error handling for API connection issues.

- [x] T023 Handle the case where the RAG agent finds no relevant content and returns an empty response.
