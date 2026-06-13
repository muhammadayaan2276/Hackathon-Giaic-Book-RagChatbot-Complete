#!/usr/bin/env python3
"""
Simple test to verify Qdrant connection with your current .env settings.
Run this first to check if your QDRANT_URL and API_KEY work.
"""

import os
from qdrant_client import QdrantClient

# Check if environment variables are set
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print("Checking QDRANT configuration...")
print(f"QDRANT_URL: {qdrant_url}")
print(f"QDRANT_API_KEY: {'***' if qdrant_api_key else 'NOT SET'}")

if not qdrant_url or not qdrant_api_key:
    print("\nERROR: Please set QDRANT_URL and QDRANT_API_KEY in your .env file")
    print("Example:")
    print("QDRANT_URL=https://your-cluster.qdrant.io")
    print("QDRANT_API_KEY=your-api-key-here")
    exit(1)

try:
    # Try to connect to Qdrant
    print("\nAttempting to connect to Qdrant...")
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=10
    )
    
    # Test connection by getting collections (this will work even if collection is empty)
    collections = client.get_collections()
    print("✅ Connection successful!")
    print(f"Available collections: {[c.name for c in collections.collections]}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nPossible issues:")
    print("- Wrong QDRANT_URL format")
    print("- Invalid API key")
    print("- Network connectivity issues")
    print("- Qdrant cluster doesn't exist anymore")
    exit(1)

print("\nTest completed successfully!")
print("Your QDRANT connection is working. Now you can run:")
print("python reingest_with_local_embeddings.py")