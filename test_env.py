import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Print Qdrant host and port
print("QDRANT_HOST:", os.getenv("QDRANT_HOST"))
print("QDRANT_PORT:", os.getenv("QDRANT_PORT"))
