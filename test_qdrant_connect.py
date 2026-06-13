import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load .env
dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
    print("Loaded .env from", dotenv_path)
else:
    print(".env not found")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print("QDRANT_URL:", qdrant_url)
print("QDRANT_API_KEY present:", bool(qdrant_api_key))

try:
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60
    )
    print("QdrantClient created successfully.")

    # Try a safe read-only operation
    collections = client.get_collections()
    print("Collections:", [c.name for c in collections.collections])

except Exception as e:
    print("Error:", type(e).__name__, ":", str(e))