import os
import json
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Force .env
dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=True)
    print("Loaded .env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "docusaurus_book"
BACKUP_FILE = f"{COLLECTION_NAME}_backup.json"

def backup():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: QDRANT_URL or QDRANT_API_KEY missing")
        return

    print(f"Backing up '{COLLECTION_NAME}'...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    points = []
    try:
        # New qdrant-client: scroll returns list of PointStruct
        batch = client.scroll(collection_name=COLLECTION_NAME, limit=1000)
        if isinstance(batch, tuple) and len(batch) == 2:
            # Older version: (points, next_offset)
            points.extend(batch[0])
        else:
            # Newer version: list of points
            points.extend(batch)
        print(f"Found {len(points)} points")
    except Exception as e:
        print(f"Scroll failed: {e}")
        return

    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump([p.dict() for p in points], f, ensure_ascii=False, indent=2)
    print(f"Backup saved: {BACKUP_FILE}")

def restore():
    if not os.path.exists(BACKUP_FILE):
        print(f"Error: {BACKUP_FILE} not found")
        return

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: QDRANT credentials missing")
        return

    print("Restoring...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted old collection")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 384, "distance": "Cosine"}
    )
    print("Created new collection")

    with open(BACKUP_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert back to PointStruct
    from qdrant_client.http import models
    points = [
        models.PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p.get("payload", {})
        ) for p in data
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch_size])
        print(f"Upserted {i+len(points[i:i+batch_size])}/{len(points)}")

    print("Restore done.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore()
    else:
        backup()