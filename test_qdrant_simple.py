
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def test_qdrant():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"Connecting to: {url}")
    
    try:
        # Try with default port 6333 explicitly or without it
        client = QdrantClient(url=url, api_key=api_key)
        collections = client.get_collections()
        print("Successfully connected!")
        print(f"Collections: {collections}")
        
        for coll in collections.collections:
            count = client.count(collection_name=coll.name)
            print(f"Collection: {coll.name}, Points: {count}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_qdrant()
