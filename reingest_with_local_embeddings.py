"""
Re-ingestion script to update the Qdrant collection with local embeddings.
This is needed because the local embedding model (all-MiniLM-L6-v2) produces 
384-dimensional vectors, while Cohere embeddings produce 1024-dimensional vectors.
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_local_embedder():
    """Create a local embedder using sentence-transformers."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Local embedding model loaded.")
    return model


def load_book_content():
    """Load the book content from the My-Book directory."""
    content_dir = Path("My-Book/docs")
    documents = []
    
    for md_file in content_dir.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Create document structure
            doc = {
                "id": str(md_file.relative_to(content_dir)).replace('/', '_').replace('\\', '_'),
                "text": content,
                "title": md_file.name,
                "url": f"file://{md_file}",
                "section": str(md_file.parent.name)
            }
            documents.append(doc)
    
    logging.info(f"Loaded {len(documents)} documents from {content_dir}")
    return documents


def reingest_to_qdrant():
    """Re-ingest documents to Qdrant using local embeddings."""
    # Initialize local embedder
    embedder = create_local_embedder()
    
    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY not set in environment")
    
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60
    )
    
    # Load documents
    documents = load_book_content()
    
    # Prepare points for Qdrant
    points = []
    for i, doc in enumerate(documents):
        # Generate embedding using local model
        embedding = embedder.encode(doc["text"]).tolist()
        
        # Create Qdrant point
        point = models.PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": doc["text"],
                "title": doc["title"],
                "url": doc["url"],
                "section": doc["section"]
            }
        )
        points.append(point)
        
        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i + 1}/{len(documents)} documents")
    
    # Get the collection name
    collection_name = "docusaurus_book"
    
    # Delete the existing collection if it exists
    try:
        client.delete_collection(collection_name)
        logging.info(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        logging.info(f"Collection {collection_name} may not exist yet: {e}")
    
    # Create a new collection with the correct vector size (384 for all-MiniLM-L6-v2)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    logging.info(f"Created new collection: {collection_name} with 384-dim vectors")
    
    # Upload points to Qdrant
    logging.info(f"Uploading {len(points)} points to Qdrant...")
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=10  # Process in batches
    )
    
    logging.info(f"Successfully re-ingested {len(documents)} documents to Qdrant collection '{collection_name}'")
    logging.info("The collection now uses local embeddings (384 dimensions) compatible with the new system.")


if __name__ == "__main__":
    print("Re-ingestion Tool for Local Embeddings")
    print("="*50)
    print("This will:")
    print("1. Load all book content from My-Book/docs/")
    print("2. Generate local embeddings using all-MiniLM-L6-v2")
    print("3. Create a new Qdrant collection with 384-dim vectors")
    print("4. Upload the embedded content to Qdrant")
    print("\nThis will DELETE the existing collection and recreate it!")
    print("="*50)

    print("Proceeding with re-ingestion...")

    try:
        reingest_to_qdrant()
        print("\nRe-ingestion completed successfully!")
        print("You can now run the RAG agent with local embeddings.")
    except Exception as e:
        print(f"Error during re-ingestion: {e}")
        sys.exit(1)