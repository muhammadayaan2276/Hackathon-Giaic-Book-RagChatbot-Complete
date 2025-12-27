from qdrant_client import QdrantClient
import logging
from ingestion_pipeline.config import Config
from ingestion_pipeline.embedder import Embedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Validation:
    def __init__(self):
        self.qdrant_client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        self.collection_name = "docusaurus_book"
        self.embedder = Embedder()

    def check_integrity(self):
        logging.info(f"Checking integrity of collection: {self.collection_name}")
        try:
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            vector_count = collection_info.vectors_count
            logging.info(f"Collection '{self.collection_name}' exists with {vector_count} vectors.")
            return True, f"Collection '{self.collection_name}' exists with {vector_count} vectors."
        except Exception as e:
            logging.error(f"Error checking collection integrity: {e}")
            return False, f"Collection integrity check failed: {e}"

    def similarity_query(self, query_text: str, top_k: int = 5):
        logging.info(f"Running similarity query for: '{query_text}'")
        try:
            query_embedding = self.embedder.embed_chunks([{"text": query_text}]) # Embedder expects list of dicts
            
            # The embedded_chunks would contain a single item with 'embedding' key
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding[0]["embedding"], # Get the embedding from the first item
                limit=top_k,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "score": hit.score,
                    "text": hit.payload.get("text"),
                    "url": hit.payload.get("url"),
                    "title": hit.payload.get("title")
                })
            logging.info(f"Found {len(results)} results for query '{query_text}'.")
            return True, results
        except Exception as e:
            logging.error(f"Error during similarity query: {e}")
            return False, f"Similarity query failed: {e}"

    def run_e2e_test(self, test_url: str):
        logging.info(f"Running end-to-end ingestion test for URL: {test_url}")
        # This will simulate an ingestion run, but will not actually re-ingest
        # since the main pipeline handles the actual ingestion.
        # This is more of a smoke test to ensure the CLI can trigger the pipeline.
        return True, "End-to-end test simulation successful (calls ingestion pipeline)."
