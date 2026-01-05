# RAG Chatbot Frontend

This is the improved frontend for the RAG Chatbot system. It connects to the backend API to provide answers based on your knowledge base.

## Features

- Modern, responsive chat interface
- Real-time conversation with the RAG system
- Text selection integration to provide context
- Auto-scrolling to latest messages
- Typing indicators for better UX
- Timestamps on messages
- Mobile-friendly design

## How to Run

1. First, make sure the backend API is running:
   ```bash
   cd hackathon-Giaic
   pip install -r requirements.txt
   uvicorn api:app --reload
   ```

2. In a new terminal, start the Docusaurus frontend:
   ```bash
   cd My-Book
   npm install
   npm start
   ```

3. Open your browser to `http://localhost:3000` and navigate to the "RAG Chatbot" page

## API Connection

The chatbot connects to the backend API at `http://localhost:8000/chat` to process queries using the RAG system. The API expects a JSON payload with:
- `query`: The user's question
- `selected_text`: Optional context from selected text on the page

## Components

- `src/components/Chatbot.js`: The main chatbot component with improved UI
- `src/components/Chatbot.module.css`: Styling for the chatbot component
- `src/pages/chatbot.js`: The page that renders the chatbot component

## Design Notes

- The chat interface follows modern messaging app conventions
- User messages appear on the right, bot responses on the left
- Messages include timestamps for better context
- The interface includes a typing indicator when waiting for responses
- Responsive design works on mobile and desktop
- Selected text from the page is automatically included as context