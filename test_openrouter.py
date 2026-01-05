"""
Test script to check if OpenRouter API is working properly
"""

import os
import httpx
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def test_openrouter():
    """Test the OpenRouter API with a simple request"""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    if not openrouter_api_key:
        print("X OPENROUTER_API_KEY not found in environment")
        return

    print("OK OPENROUTER_API_KEY found, testing API...")

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv('APP_URL', 'http://localhost'),
        "X-Title": "RAG-Book-Agent-Test"
    }

    payload = {
        "model": "mistralai/devstral-2512:free",
        "messages": [
            {"role": "user", "content": "Hello, are you working?"}
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            print(f"Response status: {res.status_code}")

            if res.status_code != 200:
                print(f"X API request failed with status {res.status_code}")
                print(f"Response: {res.text}")
                return

            response_data = res.json()
            content = response_data['choices'][0]['message']['content']
            # Remove or replace problematic Unicode characters
            content_clean = content.encode('ascii', errors='ignore').decode('ascii', errors='ignore')
            print("OK API request successful!")
            print(f"Response content: {content_clean}")

    except Exception as e:
        print(f"X Error during OpenRouter test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_openrouter())