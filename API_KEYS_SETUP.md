# API Key Setup Instructions

To use the RAG agent, you need to set up your API keys in the `.env` file:

## Required API Keys

1. **COHERE_API_KEY**: 
   - Go to https://dashboard.cohere.com/
   - Create an account and get your API key
   - Replace `your_cohere_api_key_here` with your actual key

2. **QDRANT_URL**:
   - This should be your Qdrant cloud instance URL
   - Example: `https://your-instance.us-east4-0.gcp.cloud.qdrant.io:6333`
   - Replace `your_qdrant_url_here` with your actual URL

3. **QDRANT_API_KEY**:
   - Found in your Qdrant cloud dashboard
   - Replace `your_qdrant_api_key_here` with your actual key

4. **OPENAI_API_KEY**:
   - Go to https://platform.openai.com/api-keys
   - Create an account and get your API key
   - Replace `your_openai_api_key_here` with your actual key

## Example .env file:
```
COHERE_API_KEY=your-actual-cohere-key-here
QDRANT_URL=https://your-instance.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your-actual-qdrant-key-here
OPENAI_API_KEY=your-actual-openai-key-here
```

After updating the .env file with your actual API keys, you can run the agent:
```bash
python src/retrieval/agent.py "Your question here"
```