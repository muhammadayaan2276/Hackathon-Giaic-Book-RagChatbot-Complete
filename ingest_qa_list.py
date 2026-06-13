
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_qa_list(file_path: str) -> List[Dict[str, str]]:
    """Parses the 500_qa_list.md file into a list of Q&A dicts."""
    qa_pairs = []
    current_q = None
    current_a = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split by Q[Number]:
    blocks = re.split(r'Q\d+:', content)
    
    for block in blocks:
        if not block.strip() or block.startswith('#'):
            continue
            
        parts = block.split('A:', 1)
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            qa_pairs.append({
                "question": question,
                "answer": answer
            })
            
    logging.info(f"Parsed {len(qa_pairs)} Q&A pairs from {file_path}")
    return qa_pairs

def ingest_qa():
    """Ingests Q&A pairs to Qdrant."""
    qa_file = "docs/500_qa_list.md"
    if not os.path.exists(qa_file):
        qa_file = "hackathon-Giaic/docs/500_qa_list.md"
        
    qa_pairs = parse_qa_list(qa_file)
    
    # Initialize local embedder
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    collection_name = "docusaurus_book"
    
    # Prepare points
    points = []
    for i, pair in enumerate(qa_pairs):
        # We embed the QUESTION for retrieval
        embedding = model.encode(pair["question"]).tolist()
        
        # The payload contains the ANSWER
        point = models.PointStruct(
            id=1000 + i, # Start IDs from 1000 to avoid conflict with main docs
            vector=embedding,
            payload={
                "text": pair["answer"],
                "question": pair["question"],
                "title": "500 Q&A List",
                "url": "https://hackathon-giaic-book-rag-chatbot-co.vercel.app/",
                "section": "General Q&A"
            }
        )
        points.append(point)
        
    # Upload
    logging.info(f"Uploading {len(points)} Q&A points to Qdrant...")
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=20
    )
    logging.info("Successfully ingested Q&A list!")

if __name__ == "__main__":
    try:
        ingest_qa()
    except Exception as e:
        print(f"Error: {e}")
