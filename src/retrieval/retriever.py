import os
import logging
from typing import List, Dict, Any

import cohere
from qdrant_client import QdrantClient, models

# Assuming Config is available from ingestion_pipeline
from ingestion_pipeline.config import Config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CohereEmbedder:
    """
    Handles generating embeddings for queries using the Cohere API.
    """
    def __init__(self):
        """
        Initializes the Cohere client.
        Raises:
            ValueError: If COHERE_API_KEY is not configured.
        """
        if not Config.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not configured.")
        self.co = cohere.Client(Config.COHERE_API_KEY)
        logging.info("Cohere client initialized.")

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding vector for a given query string.

        Args:
            query (str): The input query string.

        Returns:
            List[float]: The embedding vector for the query.

        Raises:
            cohere.CohereAPIError: If there's an issue with the Cohere API.
            Exception: For any other unexpected errors during embedding.
        """
        try:
            response = self.co.embed(
                texts=[query],
                model='embed-english-v3.0', # Use the same model as ingestion
                input_type='search_query'
            )
            return response.embeddings[0]
        except cohere.CohereAPIError as e:
            logging.error(f"Cohere API error during embedding: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during Cohere embedding: {e}")
            raise

class QdrantRetriever:
    """
    Handles retrieving relevant chunks from a Qdrant collection using vector search.
    """
    def __init__(self):
        """
        Initializes the Qdrant client and Cohere embedder.
        Raises:
            ValueError: If QDRANT_URL or QDRANT_API_KEY is not configured.
        """
        if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is not configured.")
        
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=60 # Use the same timeout as ingestion_pipeline
        )
        self.collection_name: str = "docusaurus_book" # From data-model.md
        self.embedder: CohereEmbedder = CohereEmbedder() # Initialize embedder here
        logging.info("Qdrant client initialized.")

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant chunks from the Qdrant collection for a given query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents
                                   a retrieved chunk with its id, score, text, url, title,
                                   section, and chunk_index. Returns an empty list if no
                                   relevant chunks are found or an error occurs.
        """
        if not query:
            logging.warning("Received empty query. Returning no results.")
            return []

        logging.info(f"Generating embeddings for query: '{query}'")
        try:
            query_vector: List[float] = self.embedder.embed_query(query)
        except Exception as e:
            logging.error(f"Failed to embed query: {e}")
            return []

        logging.info(f"Performing top-{top_k} vector search in Qdrant collection '{self.collection_name}'")
        try:
            # Using the correct Qdrant client query method for newer versions
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True, # Ensure payload is returned
                score_threshold=0.5 # Optional: filter out very low relevance results
            ).points
        except AttributeError as e:
            # Handle case where query_points method doesn't exist
            logging.error(f"Qdrant client does not have 'query_points' method: {e}")
            # Try using the legacy search method if available
            try:
                search_result: List[models.ScoredPoint] = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                    score_threshold=0.5
                )
            except AttributeError:
                # Try using the even older method if available
                try:
                    search_result: List[models.ScoredPoint] = self.client.search_points(
                        collection_name=self.collection_name,
                        vector=query_vector,
                        limit=top_k,
                        with_payload=True,
                        score_threshold=0.5
                    )
                except Exception as e3:
                    logging.error(f"Error during Qdrant search using all methods: {e3}")
                    return []
            except Exception as e2:
                logging.error(f"Error during Qdrant search using fallback method: {e2}")
                return []
        except Exception as e:
            logging.error(f"Error during Qdrant search: {e}")
            return []

        retrieved_chunks: List[Dict[str, Any]] = []
        if not search_result:
            logging.info("No relevant chunks found for the query.")
            return []

        for hit in search_result:
            chunk: Dict[str, Any] = {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text"),
                "url": hit.payload.get("url"),
                "title": hit.payload.get("title"),
                "section": hit.payload.get("section"),
                "chunk_index": hit.payload.get("chunk_index")
            }
            retrieved_chunks.append(chunk)
            logging.debug(f"Retrieved chunk (Score: {hit.score:.4f}, Title: {chunk['title']}, URL: {chunk['url']})")
        
        logging.info(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query}'")
        return retrieved_chunks