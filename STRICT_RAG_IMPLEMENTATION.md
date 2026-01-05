# Strict RAG Agent Implementation

This project implements a strict Retrieval-Augmented Generation (RAG) agent that follows specific mandatory rules to ensure accurate and reliable responses based only on provided context.

## Mandatory Rules Implemented

1. **Context-Only Responses**: The agent answers ONLY using the provided Context
2. **No External Knowledge**: The agent does NOT use its own knowledge, training data, or assumptions
3. **No Guessing**: The agent does NOT guess or hallucinate information
4. **Fallback Response**: If the answer is not in the Context, the agent replies exactly with "Answer not found in book."
5. **Concise Answers**: The agent keeps answers factual and to the point
6. **No References**: The agent does NOT mention context, book, source, or document in answers

## Files Created

- `src/retrieval/strict_rag_agent.py`: The main implementation of the strict RAG agent
- `test_strict_rag.py`: Test script to verify the agent's behavior
- `cli.py`: CLI wrapper for easy execution

## How to Use

1. Ensure you have set up your `.env` file with the required API keys:
   ```
   COHERE_API_KEY=your-actual-cohere-key-here
   QDRANT_URL=your-qdrant-url
   QDRANT_API_KEY=your-qdrant-api-key
   OPENAI_API_KEY=your-openai-api-key
   ```

2. Make sure your Qdrant instance has the book content properly indexed in a collection named "docusaurus_book"

3. Run the strict RAG agent with a query:
   ```bash
   python src/retrieval/strict_rag_agent.py "Your question about the book here"
   ```

## Features

- Strict adherence to the mandatory rules through system prompt engineering
- Proper error handling and fallback responses
- Logging for debugging and monitoring
- Integration with the existing project architecture
- Compatibility with the Cohere embedding service and Qdrant vector database

## Testing

The implementation includes:
- A test script that verifies the agent's behavior
- Proper handling of queries not found in the context
- Verification that the agent responds with "Answer not found in book." when appropriate

The strict RAG agent ensures reliable, context-bound responses while preventing hallucinations or the use of external knowledge.