#!/usr/bin/env python3
"""
Test script to verify backend components work:
- .env loading
- LocalEmbedder (local model path fallback)
- QdrantRetriever connection
- Chunk retrieval

Run with: python test_backend.py
"""

import os
from dotenv import load_dotenv
from src.retrieval.strict_rag_agent import LocalEmbedder, QdrantRetriever

# Load env
load_dotenv()
print("✅ Loaded .env")

# Test embedder
print("\n🔍 Testing LocalEmbedder...")
try:
    embedder = LocalEmbedder()
    print("✅ LocalEmbedder initialized")
except Exception as e:
    print(f"❌ LocalEmbedder failed: {e}")
    exit(1)

# Test retriever
print("\n🔍 Testing QdrantRetriever...")
try:
    retriever = QdrantRetriever(embedder)
    print("✅ QdrantRetriever initialized")
except Exception as e:
    print(f"❌ QdrantRetriever failed: {e}")
    print("💡 Check: QDRANT_URL and QDRANT_API_KEY in .env")
    exit(1)

# Test retrieval
print("\n🔍 Testing retrieve_chunks...")
try:
    chunks = retriever.retrieve_chunks("ROS 2", top_k=1)
    print(f"✅ Retrieved {len(chunks)} chunks")
    if chunks:
        print(f"  Sample chunk title: {chunks[0].get('title', 'N/A')}")
        print(f"  Sample chunk text: {chunks[0].get('text', '')[:100]}...")
except Exception as e:
    print(f"❌ retrieve_chunks failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Backend test completed.")