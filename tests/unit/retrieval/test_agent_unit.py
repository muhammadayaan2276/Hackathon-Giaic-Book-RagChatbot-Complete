import os
import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.agent import CohereEmbedder, QdrantRetriever, Config

@patch.dict(os.environ, {"COHERE_API_KEY": "test_key"})
def test_cohere_embedder_initialization():
    """Tests that CohereEmbedder initializes correctly with an API key."""
    with patch('cohere.Client') as mock_cohere_client:
        embedder = CohereEmbedder()
        mock_cohere_client.assert_called_once_with("test_key")
        assert embedder.co is not None

def test_cohere_embedder_initialization_fails_without_key():
    """Tests that CohereEmbedder raises ValueError if API key is missing."""
    with patch.dict(os.environ, {"COHERE_API_KEY": ""}):
        with pytest.raises(ValueError, match="COHERE_API_KEY is not configured."):
            CohereEmbedder()

@patch.dict(os.environ, {
    "QDRANT_URL": "http://test-qdrant.com",
    "QDRANT_API_KEY": "test_qdrant_key",
    "COHERE_API_KEY": "test_cohere_key"
})
def test_qdrant_retriever_initialization():
    """Tests that QdrantRetriever initializes correctly."""
    with patch('src.retrieval.agent.QdrantClient') as mock_qdrant_client:
        mock_embedder = MagicMock(spec=CohereEmbedder)
        retriever = QdrantRetriever(mock_embedder)
        mock_qdrant_client.assert_called_once_with(
            url="http://test-qdrant.com",
            api_key="test_qdrant_key",
            timeout=60
        )
        assert retriever.client is not None
        assert retriever.embedder is mock_embedder

@patch.dict(os.environ, {"QDRANT_URL": "", "QDRANT_API_KEY": ""})
def test_qdrant_retriever_initialization_fails_without_config():
    """Tests that QdrantRetriever raises ValueError if Qdrant config is missing."""
    with pytest.raises(ValueError, match="QDRANT_URL or QDRANT_API_KEY is not configured."):
        mock_embedder = MagicMock(spec=CohereEmbedder)
        QdrantRetriever(mock_embedder)

# More tests can be added to mock the search/embed calls
