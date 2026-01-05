# Backend Deployment Guide

This document explains how to deploy the RAG Chatbot backend to Hugging Face Spaces or other platforms.

## Backend Overview

The backend is a FastAPI application that provides a RAG (Retrieval-Augmented Generation) chatbot service. It:

- Accepts user queries via the `/chat` endpoint
- Retrieves relevant context from a Qdrant vector database
- Generates answers using an LLM via OpenRouter API
- Uses local embeddings for document processing

## Required Files for Deployment

1. `api.py` - The main FastAPI application
2. `src/` directory - Contains the RAG implementation
3. `requirements.txt` - Python dependencies
4. `Procfile` - Deployment configuration
5. Environment variables (set in deployment platform)

## Environment Variables

You need to set these environment variables in your deployment platform:

```
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Deployment to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose "Docker" or "Gradio" SDK (for FastAPI apps)
3. Add your repository files
4. Set the environment variables in the Space settings
5. The Procfile will handle starting the application

## Alternative: Deployment to other platforms

For platforms like Render, Railway, or Vercel:

1. Create an account and new project
2. Connect to your GitHub repository
3. Set build and start commands:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Set environment variables in the platform's settings

## Testing the Deployment

Once deployed, you can test the backend:

- GET `/` - Health check endpoint
- POST `/chat` - Chat endpoint (expects JSON with `query` field)

Example request:
```json
{
  "query": "What is ROS 2?"
}
```

## Troubleshooting

- If you get 500 errors, check that all environment variables are properly set
- If the app crashes on startup, verify that all dependencies are in requirements.txt
- If queries return "Answer not found in book", verify that your Qdrant database has been populated with documents