import logging
import requests 

from ingestion_pipeline.config import Config
from ingestion_pipeline.crawler import Crawler
from ingestion_pipeline.extractor import Extractor
from ingestion_pipeline.chunker import Chunker
from ingestion_pipeline.embedder import Embedder
from ingestion_pipeline.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ingestion_pipeline(url: str, rebuild: bool = False):
    # Validate configuration first
    try:
        Config.validate()
        logging.info("Configuration validated successfully!")
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        exit(1)

    logging.info(f"Starting ingestion for Docusaurus site: {url}")
    crawler = Crawler(base_url=url)
    extractor = Extractor()
    chunker = Chunker()
    embedder = Embedder()
    vector_store = VectorStore()

    if rebuild:
        logging.info(f"Rebuilding Qdrant collection '{vector_store.collection_name}'.")
        vector_store.recreate_collection() # Assuming VectorStore has this method, will add later

    crawled_urls = crawler.crawl()
    
    extracted_data = []
    for page_url in crawled_urls:
        try:
            response = requests.get(page_url)
            response.raise_for_status()
            extracted = extractor.extract_content(response.text)
            extracted_data.append({
                "url": page_url,
                "title": extracted["title"],
                "content": extracted["content"]
            })
            logging.info(f"Extracted content from: {page_url}")
            logging.debug(f"Content for {page_url}: \n{extracted['content'][:500]}...") # Log first 500 chars of content
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching/extracting content from {page_url}: {e}")
        except Exception as e:
            logging.error(f"Error processing {page_url}: {e}")
    
    logging.info(f"Finished extracting content from {len(extracted_data)} pages.")

    if extracted_data:
        chunks = chunker.chunk_text(extracted_data)
        logging.info(f"Created {len(chunks)} chunks.")
        
        try:
            embedded_chunks = embedder.embed_chunks(chunks)
            logging.info(f"Generated embeddings for {len(embedded_chunks)} chunks.")
            
            vector_store.upsert_chunks(embedded_chunks)
            logging.info(f"Successfully upserted {len(embedded_chunks)} chunks into Qdrant.")
            logging.info(f"Ingestion summary: {len(crawled_urls)} URLs crawled, {len(extracted_data)} pages extracted, {len(chunks)} chunks, {len(embedded_chunks)} vectors upserted.")

        except Exception as e:
            logging.error(f"Error during embedding or vector storage: {e}")
            exit(1)

    else:
        logging.info("No extracted data found to process.")

if __name__ == "__main__":
    # This block will no longer be executed directly as cli.py is the new entry point
    # but keeping for completeness if someone tries to run pipeline.py directly without args
    logging.warning("pipeline.py is intended to be run via cli.py. No action performed when run directly.")

