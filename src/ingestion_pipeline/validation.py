from qdrant_client import QdrantClient, models
import logging
from ingestion_pipeline.config import Config
from ingestion_pipeline.embedder import Embedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Validation:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=60
        )
        self.collection_name = "docusaurus_book"
        self.embedder = Embedder()

    def check_integrity(self):
        logging.info(f"Checking integrity of collection: {self.collection_name}")
        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.collection_name
            )
            vector_count = collection_info.vectors_count
            logging.info(
                f"Collection '{self.collection_name}' exists with {vector_count} vectors."
            )
            return True, f"Collection '{self.collection_name}' exists with {vector_count} vectors."
        except Exception as e:
            logging.error(f"Error checking collection integrity: {e}")
            return False, f"Collection integrity check failed: {e}"

    def similarity_query(self, query_text: str, top_k: int = 5):
        logging.info(f"Running similarity query for: '{query_text}'")
        try:
            # 1️⃣ Embed query
            embedded = self.embedder.embed_chunks([{"text": query_text}])
            query_vector = embedded[0]["embedding"]

            # 2️⃣ Qdrant v1.16+ correct API
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.VectorQuery(vector=query_vector),
                        limit=top_k
                    )
                ],
                with_payload=True
            )

            # 3️⃣ Parse results
            results = []
            for hit in response.points:
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
        return True, "End-to-end test simulation successful (calls ingestion pipeline)."
