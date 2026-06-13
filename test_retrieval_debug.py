
import os
import asyncio
from src.retrieval.strict_rag_agent import LocalEmbedder, QdrantRetriever, StrictRAGAgent
from dotenv import load_dotenv

load_dotenv()

async def debug_retrieval():
    embedder = LocalEmbedder()
    retriever = QdrantRetriever(embedder)
    
    queries = [
        "Chapter 3: Motion Planning Basics",
        "chapter 4",
        "what are you doing?"
    ]
    
    for query in queries:
        print(f"\n--- Testing Query: '{query}' ---")
        chunks = retriever.retrieve_chunks(query, top_k=3)
        if not chunks:
            print("No chunks retrieved.")
            continue
            
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} (Score: {chunk['score']:.4f}):")
            print(f"Title: {chunk.get('title')}")
            print(f"Section: {chunk.get('section')}")
            # Print first 100 chars of text
            text = chunk.get('text', '')
            print(f"Text snippet: {text[:200]}...")
            print("-" * 20)

if __name__ == "__main__":
    asyncio.run(debug_retrieval())
