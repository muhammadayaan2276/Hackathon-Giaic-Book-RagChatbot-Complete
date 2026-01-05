# API Key Setup Instructions (Updated)

To use the FREE RAG agent, you need to set up your API keys in the `.env` file:

## Required API Keys

1. **OPENROUTER_API_KEY**:
   - Go to https://openrouter.ai/keys
   - Create an account and get your API key
   - Replace `your_openrouter_api_key_here` with your actual key

2. **QDRANT_URL**:
   - This should be your Qdrant cloud instance URL
   - Example: `https://your-instance.us-east4-0.gcp.cloud.qdrant.io:6333`
   - Replace `your_qdrant_url_here` with your actual URL

3. **QDRANT_API_KEY**:
   - Found in your Qdrant cloud dashboard
   - Replace `your_qdrant_api_key_here` with your actual key

## Example .env file:
```
OPENROUTER_API_KEY=your-actual-openrouter-key-here
QDRANT_URL=https://your-instance.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your-actual-qdrant-key-here
```

## Important Notes:
- NO COHERE API KEY REQUIRED
- NO OPENAI API KEY REQUIRED
- The system now uses local embeddings (sentence-transformers)
- The system now uses OpenRouter with a free model (mistralai/devstral-2512:free)

After updating the .env file with your actual API keys, you can run the agent:
```bash
python src/retrieval/strict_rag_agent.py "Your question here"
```