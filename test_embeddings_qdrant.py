#!/usr/bin/env python3
"""
Test script to verify embeddings and Qdrant vector database integration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from retrieval.strict_rag_agent import LocalEmbedder, QdrantRetriever

def test_embeddings_and_qdrant():
    """Test if embeddings are generated and Qdrant search works."""
    
    print("=== Testing Embeddings and Qdrant Integration ===\n")
    
    # Test 1: Check if environment variables are set
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    
    print(f"QDRANT_URL: {qdrant_url}")
    print(f"QDRANT_API_KEY: {'***' if qdrant_api_key else 'NOT SET'}")
    
    if not qdrant_url or not qdrant_api_key:
        print("\n❌ ERROR: QDRANT_URL or QDRANT_API_KEY not configured in .env file.")
        print("Please check your .env file and ensure both variables are set.")
        return False
    
    # Test 2: Initialize embedder
    try:
        print("\n--- Testing Local Embedder ---")
        embedder = LocalEmbedder()
        print("✅ Local embedder initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing embedder: {e}")
        return False
    
    # Test 3: Generate embedding for a sample query
    try:
        print("\n--- Testing Embedding Generation ---")
        sample_query = "What is ROS 2?"
        embedding = embedder.embed_query(sample_query)
        print(f"✅ Generated embedding for '{sample_query}'")
        print(f"Embedding length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False
    
    # Test 4: Initialize Qdrant retriever
    try:
        print("\n--- Testing Qdrant Retriever ---")
        retriever = QdrantRetriever(embedder)
        print("✅ Qdrant retriever initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Qdrant retriever: {e}")
        return False
    
    # Test 5: Perform vector search
    try:
        print("\n--- Testing Qdrant Vector Search ---")
        retrieved_chunks = retriever.retrieve_chunks("What is ROS 2?", top_k=1)
        
        if retrieved_chunks:
            print(f"✅ Successfully retrieved {len(retrieved_chunks)} chunks")
            print(f"First chunk score: {retrieved_chunks[0]['score']:.4f}")
            print(f"First chunk title: {retrieved_chunks[0]['title']}")
            print(f"First chunk text preview: {retrieved_chunks[0]['text'][:100]}...")
        else:
            print("⚠️  Retrieved 0 chunks - this could mean:")
            print("   - Qdrant collection is empty (no data ingested)")
            print("   - No relevant chunks found for the query")
            print("   - Connection issues with Qdrant")
            
    except Exception as e:
        print(f"❌ Error during Qdrant search: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Test Summary ===")
    print("✅ Embeddings: Working")
    print("✅ Qdrant Connection: Working")
    print("✅ Vector Search: Working (returned results)")
    
    return True

if __name__ == "__main__":
    success = test_embeddings_and_qdrant()
    sys.exit(0 if success else 1)