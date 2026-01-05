import os
from qdrant_client import QdrantClient
import qdrant_client.qdrant_client
from agents import Agent # Importing from openai-agents to see if it causes a conflict

def main():
    print(f"Qdrant client version: {qdrant_client.__version__}")

    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    if not qdrant_url or not qdrant_api_key:
        print("QDRANT_URL or QDRANT_API_KEY is not configured.")
        return

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60
    )

    print(f"QdrantClient object: {client}")
    print(f"Has 'search' attribute: {hasattr(client, 'search')}")
    print(f"Has 'query' attribute: {hasattr(client, 'query')}")


    if not hasattr(client, 'search'):
        print("The 'search' method is missing from the QdrantClient object.")
        print("Available attributes:")
        print(dir(client))

if __name__ == "__main__":
    # Load .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        with open(dotenv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    if os.getenv(key) is None:
                        os.environ[key] = value
    main()