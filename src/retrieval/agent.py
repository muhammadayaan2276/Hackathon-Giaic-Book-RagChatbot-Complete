# import typer
# import os
# import sys
# from pathlib import Path
# import cohere
# from typing import List, Dict, Any
# from qdrant_client import QdrantClient
# import logging
# import asyncio
# from agents import Agent, Runner
# from agents import OpenAIChatCompletionsModel
# from openai import AsyncOpenAI

# # Add the src directory to the Python path to allow imports
# sys.path.append(str(Path(__file__).parent.parent))

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Import Config from ingestion pipeline to ensure consistent configuration handling
# from ingestion_pipeline.config import Config


# class CohereEmbedder:
#     """Handles generating embeddings for queries using the Cohere API."""
#     def __init__(self):
#         # Load environment variables (this will use mocked values during tests)
#         cohere_api_key = os.getenv("COHERE_API_KEY", "")
#         if not cohere_api_key:
#             raise ValueError("COHERE_API_KEY is not configured. Please set up your .env file with a valid COHERE_API_KEY. See API_KEYS_SETUP.md for instructions.")
#         self.co = cohere.Client(cohere_api_key)
#         logging.info("Cohere client initialized.")

#     def embed_query(self, query: str) -> List[float]:
#         """Generates an embedding for a given query."""
#         try:
#             response = self.co.embed(
#                 texts=[query],
#                 model='embed-english-v3.0',
#                 input_type='search_query'
#             )
#             logging.info("Successfully generated query embedding.")
#             return response.embeddings[0]
#         except Exception as e:
#             logging.error(f"Error generating Cohere embedding: {e}")
#             raise


# class QdrantRetriever:
#     """Handles retrieving relevant chunks from a Qdrant collection."""
#     def __init__(self, embedder: CohereEmbedder, client=None):
#         qdrant_url = os.getenv("QDRANT_URL", "")
#         qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
#         if not qdrant_url or not qdrant_api_key:
#             raise ValueError("QDRANT_URL or QDRANT_API_KEY is not configured. Please set up your .env file with valid QDRANT_URL and QDRANT_API_KEY. See API_KEYS_SETUP.md for instructions.")

#         if client is not None:
#             # Use provided client (for testing purposes)
#             self.client = client
#         else:
#             self.client = QdrantClient(
#                 url=qdrant_url,
#                 api_key=qdrant_api_key,
#                 timeout=60
#             )
#         self.collection_name = "docusaurus_book"
#         self.embedder = embedder
#         logging.info("Qdrant client initialized.")

#     def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Retrieves top-k relevant chunks from Qdrant."""
#         logging.info(f"Generating embeddings for query: '{query}'")
#         query_vector = self.embedder.embed_query(query)

#         logging.info(f"Performing top-{top_k} vector search in Qdrant collection '{self.collection_name}'")
#         try:
#             # Using the correct Qdrant client query method for newer versions
#             search_result = self.client.query_points(
#                 collection_name=self.collection_name,
#                 query=query_vector,
#                 limit=top_k,
#                 with_payload=True,  # Ensure payload is returned
#                 score_threshold=0.5  # Optional: filter out very low relevance results
#             ).points
#         except AttributeError as e:
#             # Handle case where query_points method doesn't exist
#             logging.error(f"Qdrant client does not have 'query_points' method: {e}")
#             # Try using the legacy search method if available
#             try:
#                 search_result = self.client.search(
#                     collection_name=self.collection_name,
#                     query_vector=query_vector,
#                     limit=top_k,
#                     with_payload=True,
#                     score_threshold=0.5
#                 )
#             except AttributeError:
#                 # Try using the even older method if available
#                 try:
#                     search_result = self.client.search_points(
#                         collection_name=self.collection_name,
#                         vector=query_vector,
#                         limit=top_k,
#                         with_payload=True,
#                         score_threshold=0.5
#                     )
#                 except Exception as e3:
#                     logging.error(f"Error during Qdrant search using all methods: {e3}")
#                     return []
#             except Exception as e2:
#                 logging.error(f"Error during Qdrant search using fallback method: {e2}")
#                 return []
#         except Exception as e:
#             logging.error(f"Error during Qdrant search: {e}")
#             return []

#         retrieved_chunks = []
#         for hit in search_result:
#             chunk = {
#                 "id": hit.id,
#                 "score": hit.score,
#                 "text": hit.payload.get("text"),
#                 "url": hit.payload.get("url"),
#                 "title": hit.payload.get("title"),
#                 "section": hit.payload.get("section"),
#                 "chunk_index": hit.payload.get("chunk_index")
#             }
#             retrieved_chunks.append(chunk)

#         logging.info(f"Retrieved {len(retrieved_chunks)} chunks for query.")
#         return retrieved_chunks


# class RAGAgent:
#     """The RAG agent that generates answers based on retrieved context."""
#     def __init__(self):
#         self.system_prompt = """
#         You are an expert assistant for a technical book about robotics and AI.
#         You MUST answer questions strictly based on the provided context.
#         Do NOT use any external knowledge.
#         If the context does not contain the answer, you MUST state that you cannot answer the question based on the provided information.
#         Cite the sources from the context, including the title and URL.
#         Provide clear, concise, and accurate answers based on the book content.
#         """
#         # Check if OpenAI API key is available
#         openai_api_key = os.getenv("OPENAI_API_KEY", "")
#         if not openai_api_key:
#             logging.warning("OPENAI_API_KEY is not configured. Please set up your .env file with a valid OPENAI_API_KEY. See API_KEYS_SETUP.md for instructions.")

#         self.agent = Agent(
#             name="RAGExpert",
#             instructions=self.system_prompt,
#             model="gpt-4o"  # Using a standard OpenAI model
#         )
#         logging.info("RAGAgent initialized.")

#     async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
#         """Generates an answer based on the query and context."""

#         if not context_chunks:
#             logging.warning("No context chunks provided. Returning fallback message.")
#             return f"I could not find any information about '{query}' in the provided book content. My knowledge is strictly limited to the technical book."

#         # Build context string with proper formatting
#         context_str = "Context:\n"
#         for i, chunk in enumerate(context_chunks):
#             # Clean up text to avoid encoding issues
#             text_content = chunk['text'] or ""
#             # Remove problematic characters
#             text_content = text_content.replace('\u200b', '')  # Zero-width space
#             context_str += f"Source {i+1} (Title: {chunk['title']}, URL: {chunk['url']}):\n"
#             context_str += f"{text_content}\n\n"

#         user_prompt = f"Question: {query}"

#         full_prompt = context_str + user_prompt

#         logging.info("Generating answer using OpenAI Agents SDK.")
        
#         try:
#             result = await Runner.run(self.agent, full_prompt)
#             answer = result.final_output
#             logging.info("Successfully generated answer.")
#             return answer
#         except Exception as e:
#             logging.error(f"Error generating OpenAI completion: {e}")
#             return "Sorry, I encountered an error while generating the answer. Please try again later."


# app = typer.Typer()


# async def _run_agent(query: str):
#     """
#     Async function to run the RAG agent.
#     """
#     logging.info(f"Received query: '{query}'")

#     try:
#         # Only load environment variables from .env file if not already set in environment
#         # This allows tests to override environment variables
#         dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
#         if os.path.exists(dotenv_path):
#             with open(dotenv_path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     line = line.strip()
#                     if line and not line.startswith('#'):
#                         key, value = line.split('=', 1)
#                         # Remove quotes if present
#                         if value.startswith('"') and value.endswith('"'):
#                             value = value[1:-1]
#                         # Only set the environment variable if it's not already set
#                         if os.getenv(key) is None:
#                             os.environ[key] = value

#         embedder = CohereEmbedder()
#         retriever = QdrantRetriever(embedder)
#         agent = RAGAgent()

#         retrieved_chunks = retriever.retrieve_chunks(query)
#         answer = await agent.generate_answer(query, retrieved_chunks)

#         print("\n--- Answer ---\n")
#         print(answer)
#         print("\n--------------\n")

#     except ValueError as e:
#         logging.error(f"Configuration Error: {e}")
#         print(f"Configuration Error: {e}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         print(f"An unexpected error occurred: {e}")


# @app.command()
# def main(query: str):
#     """
#     Retrieval-Augmented Generation Agent CLI
#     """
#     asyncio.run(_run_agent(query))


# if __name__ == "__main__":
#     app()