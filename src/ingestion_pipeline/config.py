import os

class Config:
    # These will be set within the validate method
    COHERE_API_KEY: str = ""
    QDRANT_HOST: str = ""
    QDRANT_PORT: int = 0
    QDRANT_URL: str = "" # Add QDRANT_URL as a class variable for completeness
    QDRANT_API_KEY: str = ""

    @classmethod
    def validate(cls):
        # Manually load environment variables from .env file
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        if os.path.exists(dotenv_path):
            with open(dotenv_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        os.environ[key] = value

        # Assign values to class variables after loading
        cls.COHERE_API_KEY = os.getenv("COHERE_API_KEY", "") # Default to empty string if not found
        cls.QDRANT_URL = os.getenv("QDRANT_URL", "") # Default to empty string if not found
        cls.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
        
        # Initialize QDRANT_HOST and QDRANT_PORT based on QDRANT_URL or direct env vars
        if cls.QDRANT_URL:
            from urllib.parse import urlparse
            parsed_url = urlparse(cls.QDRANT_URL)
            cls.QDRANT_HOST = parsed_url.hostname if parsed_url.hostname else ""
            # Default to 443 for https, 80 for http if port is not specified in URL
            cls.QDRANT_PORT = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)
        else:
            cls.QDRANT_HOST = os.getenv("QDRANT_HOST", "") # Fallback to direct env var
            cls.QDRANT_PORT = int(os.getenv("QDRANT_PORT", 0)) # Fallback to direct env var, convert to int

        missing_vars = []
        if not cls.COHERE_API_KEY:
            missing_vars.append("COHERE_API_KEY")
        if not cls.QDRANT_HOST:
            missing_vars.append("QDRANT_HOST")
        if not cls.QDRANT_PORT:
            missing_vars.append("QDRANT_PORT")
        if not cls.QDRANT_API_KEY:
            missing_vars.append("QDRANT_API_KEY")
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
