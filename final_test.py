#!/usr/bin/env python3
"""
Final simple test - no Unicode, minimal dependencies.
"""

import os

print("Checking your setup...")

# Check .env file
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print("QDRANT_URL:", qdrant_url)
print("QDRANT_API_KEY:", "***" if qdrant_api_key else "NOT SET")

if qdrant_url and qdrant_api_key:
    print("\nYour QDRANT credentials are configured.")
    print("To run the embedding pipeline:")
    print("1. Fix pydantic version issue first:")
    print("   pip install pydantic==2.5.3 pydantic-core==2.14.6 qdrant-client==1.8.0")
    print("2. Then run:")
    print("   python reingest_with_local_embeddings.py")
else:
    print("\nPlease update your .env file with new QDRANT credentials.")

print("\nFiles ready:")
print("- reingest_with_local_embeddings.py (embedding pipeline)")
print("- api.py (your chatbot API)")
print("- .env (configuration)")