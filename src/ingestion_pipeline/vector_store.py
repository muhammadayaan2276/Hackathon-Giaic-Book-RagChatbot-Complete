from qdrant_client import QdrantClient, models
import logging
from typing import List, Dict

from ingestion_pipeline.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        self.collection_name = "docusaurus_book" # From data-model.md

    def recreate_collection(self, vector_size: int = None):
        if vector_size is None:
            # Attempt to get existing collection info to infer vector_size if it exists
            try:
                collection_info = self.client.get_collection(collection_name=self.collection_name)
                vector_size = collection_info.config.params.vectors.size
                logging.info(f"Inferred vector size {vector_size} from existing collection.")
            except Exception:
                # Default size if collection doesn't exist or size cannot be inferred
                # This should ideally come from the embedder model
                vector_size = 1024 # Cohere embed-english-v3.0 default output dimension
                logging.warning(f"Could not infer vector size. Using default: {vector_size}. Ensure this matches your embedder.")

        logging.info(f"Recreating collection '{self.collection_name}' with vector size {vector_size} and Cosine distance.")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        logging.info(f"Collection '{self.collection_name}' recreated.")

    def create_collection_if_not_exists(self, vector_size: int):
        try:
            self.client.get_collection(collection_name=self.collection_name)
            logging.info(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            logging.info(f"Collection '{self.collection_name}' does not exist. Creating it.")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            logging.info(f"Collection '{self.collection_name}' created with vector size {vector_size} and Cosine distance.")

    def upsert_chunks(self, embedded_chunks: List[Dict]):
        if not embedded_chunks:
            logging.warning("No embedded chunks to upsert.")
            return

        # Ensure collection exists and has correct vector size
        # Assuming all embeddings have the same size
        vector_size = len(embedded_chunks[0]["embedding"])
        self.create_collection_if_not_exists(vector_size)

        points = []
        for chunk in embedded_chunks:
            points.append(
                models.PointStruct(
                    id=chunk["id"], # Using the UUID generated in Chunker as deterministic ID
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        "url": chunk["url"],
                        "title": chunk["title"],
                        "section": chunk["section"],
                        "chunk_index": chunk["chunk_index"]
                    }
                )
            )
        
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
        logging.info(f"Upserted {len(points)} points to Qdrant. Status: {operation_info.status}")
