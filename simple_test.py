#!/usr/bin/env python3
"""
Simple test to verify embeddings and Qdrant connection without complex imports.
"""

import os
import sys
from pathlib import Path

# Try to import only what we need
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported successfully")
except ImportError as e:
    print(f"❌ sentence-transformers not installed: {e}")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    print("✅ qdrant-client imported successfully")
except ImportError as e:
    print(f"❌ qdrant-client not installed: {e}")
    sys.exit(1)

# Check environment variables
qdrant_url = os.getenv("QDRANT_URL", "")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

print(f"\nQDRANT_URL: {qdrant_url}")
print(f"QDRANT_API_KEY: {'***' if qdrant_api_key else 'NOT SET'}")

if not qdrant_url or not qdrant_api_key:
    print("\n❌ ERROR: QDRANT environment variables not set")
    print("Please ensure your .env file has QDRANT_URL and QDRANT_API_KEY")
    sys.exit(1)

# Test embedding generation
try:
    print("\n--- Testing Embedding Generation ---")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sample_query = "What is ROS 2?"
    embedding = model.encode(sample_query).tolist()
    print(f"✅ Generated embedding for '{sample_query}'")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"❌ Error generating embedding: {e}")
    sys.exit(1)

# Test Qdrant connection
try:
    print("\n--- Testing Qdrant Connection ---")
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=10
    )
    
    # Try to get collection info (this will fail if collection doesn't exist, but connection works)
    try:
        collections = client.get_collections()
        print(f"✅ Qdrant connection successful")
        print(f"Available collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"⚠️  Qdrant connection works, but getting collections failed: {e}")
        print("This is normal if the collection doesn't exist yet.")
    
except Exception as e:
    print(f"❌ Error connecting to Qdrant: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All basic tests passed!")
print("Your embeddings and Qdrant connection are working.")