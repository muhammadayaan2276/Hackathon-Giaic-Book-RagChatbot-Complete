# Quickstart: Backend-Frontend Integration

**Version**: 1.0
**Status**: DRAFT
**Author**: Gemini Agent
**Created**: 2025-12-31
**Last Updated**: 2025-12-31

## 1. Backend Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r api_requirements.txt
    ```
2.  **Set Environment Variables**:
    -   Create a `.env` file in the root of the project.
    -   Add the required API keys for the LLM and Qdrant.
3.  **Run Backend Server**:
    ```bash
    uvicorn api:app --reload
    ```
    The backend will be available at `http://localhost:8000`.

## 2. Frontend Setup

1.  **Navigate to Book Directory**:
    ```bash
    cd My-Book
    ```
2.  **Install Dependencies**:
    ```bash
    npm install
    ```
3.  **Run Frontend Server**:
    ```bash
    npm start
    ```
    The book will be available at `http://localhost:3000`.

## 3. Testing

1.  Open the book at `http://localhost:3000`.
2.  Open the chatbot UI.
3.  Ask a question and verify that you get a response.
