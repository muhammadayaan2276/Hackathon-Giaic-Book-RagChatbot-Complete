import os
from dotenv import load_dotenv

load_dotenv() # take environment variables from .env.

class Config:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = os.getenv("QDRANT_PORT")

    @classmethod
    def validate(cls):
        missing_vars = []
        if not cls.COHERE_API_KEY:
            missing_vars.append("COHERE_API_KEY")
        if not cls.QDRANT_HOST:
            missing_vars.append("QDRANT_HOST")
        if not cls.QDRANT_PORT:
            missing_vars.append("QDRANT_PORT")
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
