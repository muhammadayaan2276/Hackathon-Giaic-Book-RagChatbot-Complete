import cohere
import logging
import time

from typing import List, Dict
from ingestion_pipeline.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Embedder:
    def __init__(self):
        self.co = cohere.Client(Config.COHERE_API_KEY)
        self.model = "embed-english-v3.0" # or "embed-multilingual-v3.0" for multilingual content
        self.input_type = "search_document" # or "search_query"

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []
        batch_size = 96 # Cohere recommended batch size for embed-english-v3.0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = self.co.embed(
                    texts=batch_texts,
                    model=self.model,
                    input_type=self.input_type
                ).embeddings
                embeddings.extend(response)
                logging.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except cohere.CohereAPIError as e:
                logging.error(f"Cohere API error during embedding: {e}")
                logging.warning("Retrying after 5 seconds...")
                time.sleep(5)
                try:
                    response = self.co.embed(
                        texts=batch_texts,
                        model=self.model,
                        input_type=self.input_type
                    ).embeddings
                    embeddings.extend(response)
                    logging.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} after retry.")
                except cohere.CohereAPIError as retry_e:
                    logging.error(f"Cohere API error after retry: {retry_e}")
                    # Handle more robustly, e.g., skip batch or raise
                    raise retry_e
            except Exception as e:
                logging.error(f"An unexpected error occurred during embedding: {e}")
                raise e

        # Attach embeddings back to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
        return chunks
