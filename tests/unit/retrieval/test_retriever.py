import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.retriever import CohereEmbedder, QdrantRetriever
from ingestion_pipeline.config import Config

# Mock Config for tests
@pytest.fixture(autouse=True)
def mock_config():
    with patch('ingestion_pipeline.config.Config.COHERE_API_KEY', 'test_cohere_key'), \
         patch('ingestion_pipeline.config.Config.QDRANT_URL', 'http://test_qdrant_url'), \
         patch('ingestion_pipeline.config.Config.QDRANT_API_KEY', 'test_qdrant_api_key'):
        yield

def test_cohere_embedder_initialization():
    embedder = CohereEmbedder()
    assert embedder.co is not None

@patch('cohere.Client')
def test_cohere_embedder_embed_query(mock_cohere_client):
    mock_cohere_instance = MagicMock()
    mock_cohere_client.return_value = mock_cohere_instance
    
    # Mock the response from cohere.embed
    mock_cohere_instance.embed.return_value = MagicMock(
        embeddings=[[0.1, 0.2, 0.3]]
    )

    embedder = CohereEmbedder()
    query_vector = embedder.embed_query("test query")

    mock_cohere_instance.embed.assert_called_once_with(
        texts=["test query"],
        model='embed-english-v3.0',
        input_type='search_query'
    )
    assert query_vector == [0.1, 0.2, 0.3]

def test_qdrant_retriever_initialization():
    retriever = QdrantRetriever()
    assert retriever.client is not None
    assert retriever.collection_name == "docusaurus_book"

@patch('src.retrieval.retriever.QdrantClient')
def test_qdrant_retriever_retrieve_chunks_placeholder(mock_qdrant_client):
    # This test primarily ensures the method exists and can be called without error
    # Actual search logic will be tested after implementation
    mock_qdrant_instance = MagicMock()
    mock_qdrant_client.return_value = mock_qdrant_instance

    retriever = QdrantRetriever()
    # Mocking collection info as retrieve_chunks might call create_collection_if_not_exists
    mock_qdrant_instance.get_collection.side_effect = Exception("Collection not found for test")
    mock_qdrant_instance.recreate_collection.return_value = None

    query = "test query"
    chunks = retriever.retrieve_chunks(query, top_k=5)

    assert chunks == [] # Expecting empty list as it's a placeholder
    mock_qdrant_client.assert_called_once() # Ensure client was initialized
