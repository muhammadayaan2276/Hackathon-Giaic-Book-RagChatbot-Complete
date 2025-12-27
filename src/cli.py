import argparse
import logging
from ingestion_pipeline.pipeline import run_ingestion_pipeline
from ingestion_pipeline.validation import Validation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cli_main():
    parser = argparse.ArgumentParser(description="CLI for RAG Ingestion Pipeline.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest content from a Docusaurus site.")
    ingest_parser.add_argument("--url", type=str, required=True, help="Base URL of the Docusaurus site to crawl.")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Rebuild the Qdrant collection before ingestion.")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate the ingested data.")
    validate_parser.add_argument("--query", type=str, help="Run a sample similarity query.")
    validate_parser.add_argument("--test-url", type=str, help="Run an end-to-end ingestion test with a specified URL.")
    validate_parser.add_argument("--check-integrity", action="store_true", help="Verify stored vectors and metadata correctness.")
    validate_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return for similarity query.")

    args = parser.parse_args()

    validation_service = Validation()

    if args.command == "ingest":
        logging.info(f"CLI: Running ingest command for URL: {args.url}")
        run_ingestion_pipeline(args.url, args.rebuild)
    elif args.command == "validate":
        logging.info("CLI: Running validate command.")
        if args.query:
            success, results = validation_service.similarity_query(args.query, args.top_k)
            if success:
                logging.info(f"Similarity Query Results for '{args.query}':")
                for result in results:
                    logging.info(f"  Score: {result['score']:.4f}, Title: {result['title']}, URL: {result['url']}")
            else:
                logging.error(f"Similarity query failed: {results}")
        elif args.test_url:
            success, message = validation_service.run_e2e_test(args.test_url)
            if success:
                logging.info(f"End-to-end test for {args.test_url} initiated. Message: {message}")
                # To actually run the ingestion for the test, uncomment the line below
                # run_ingestion_pipeline(args.test_url, rebuild=True) 
            else:
                logging.error(f"End-to-end test failed: {message}")
        elif args.check_integrity:
            success, message = validation_service.check_integrity()
            if success:
                logging.info(f"Integrity check passed: {message}")
            else:
                logging.error(f"Integrity check failed: {message}")
        else:
            logging.warning("CLI: No validation action specified. Use --query, --test-url, or --check-integrity.")
    else:
        parser.print_help()

if __name__ == "__main__":
    cli_main()
