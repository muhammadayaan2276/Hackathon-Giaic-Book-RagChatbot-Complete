# Technical Plan: Backend-Frontend Integration for RAG Chatbot

**Version**: 1.0
**Status**: DRAFT
**Author**: Gemini Agent
**Created**: 2025-12-31
**Last Updated**: 2025-12-31

## 1. Objective

Integrate a FastAPI backend with a Docusaurus-based book website and build a chatbot UI that uses a strict RAG agent to answer questions from book content.

## 2. Technical Context

-   **Frontend Framework**: Docusaurus (React) located in `My-Book/`.
-   **Backend Framework**: FastAPI (Python).
-   **RAG Agent**: Existing logic in `strict_rag_agent.py`.
-   **Deployment**: Local development and testing environment only.

## 3. Plan

### Phase 1: Backend API Development

1.  **Create `api.py`**:
    -   Initialize a new FastAPI application.
    -   Location: `api.py` at the root of the project.
2.  **Enable CORS**:
    -   Configure CORS middleware in `api.py` to allow requests from the Docusaurus frontend (typically `http://localhost:3000`).
3.  **Implement `/chat` Endpoint**:
    -   Create a POST endpoint at `/chat`.
    -   It will accept a JSON payload with `query` (string) and optional `selected_text` (string).
    -   This endpoint will house the RAG integration logic.

### Phase 2: RAG Integration

1.  **Integrate RAG Agent**:
    -   Import and use the agent from `strict_rag_agent.py` within the `/chat` endpoint.
    -   The agent will be responsible for handling the retrieval and generation process.
2.  **Handle Input**:
    -   The endpoint will pass the `query` and `selected_text` to the RAG agent.
3.  **Return Response**:
    -   The agent's response, which is strictly based on book content, will be returned as a JSON response from the endpoint.

### Phase 3: Chatbot UI Development (Docusaurus)

1.  **Build Chatbot Component**:
    -   Create a new React component for the chatbot UI within the `My-Book/src/components/` directory.
    -   This component will manage the chat interface, including messages and user input.
2.  **State Management**:
    -   Implement state to handle the list of messages, the current user input, and the loading status.
3.  **Query Modes**:
    -   The UI will support:
        -   A general input for full-book queries.
        -   A mechanism to capture selected text and trigger a contextual query.

### Phase 4: Frontend-Backend Integration

1.  **Connect to API**:
    -   Use the `fetch` API or a library like `axios` in the chatbot component to make POST requests to the `http://localhost:8000/chat` endpoint.
2.  **Display Data**:
    -   On receiving a response, the UI will update to display the assistant's message.
    -   A loading indicator will be shown while waiting for the response.

### Phase 5: Validation

1.  **Full-Book Queries**:
    -   Test that general questions receive relevant, book-grounded answers.
2.  **Selected-Text Queries**:
    -   Verify that questions about a specific text selection receive contextually appropriate answers.
3.  **Fallback Mechanism**:
    -   Ensure that the UI displays a helpful message when the RAG agent cannot find a relevant answer in the book content.

## 4. Deliverables

-   `api.py`: The new FastAPI backend file.
-   **Chatbot UI Component**: A new React component integrated into the Docusaurus site.
-   A fully functional, local-only integration between the frontend and backend.
