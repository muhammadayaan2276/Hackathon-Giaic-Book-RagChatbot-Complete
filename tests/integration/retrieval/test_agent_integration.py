import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from src.retrieval.agent import app

runner = CliRunner()

@patch('src.retrieval.agent.CohereEmbedder')
@patch('src.retrieval.agent.QdrantRetriever')
@patch('src.retrieval.agent.RAGAgent')
def test_main_cli_flow(MockRAGAgent, MockQdrantRetriever, MockCohereEmbedder):
    """
    Tests the main CLI flow from query to answer, mocking all external services.
    """
    # Arrange
    mock_embedder_instance = MockCohereEmbedder.return_value
    mock_retriever_instance = MockQdrantRetriever.return_value
    mock_agent_instance = MockRAGAgent.return_value

    mock_retriever_instance.retrieve_chunks.return_value = [
        {"id": "1", "score": 0.9, "text": "This is a test chunk.", "url": "/test", "title": "Test", "section": "A", "chunk_index": 0}
    ]
    from unittest.mock import AsyncMock
    mock_agent_instance.generate_answer = AsyncMock(return_value="This is a test answer.")

    # Act
    result = runner.invoke(app, ["What is a test?"])

    # Assert
    assert result.exit_code == 0
    assert "This is a test answer." in result.stdout
    mock_retriever_instance.retrieve_chunks.assert_called_once_with("What is a test?")
    mock_agent_instance.generate_answer.assert_called_once()

def test_main_cli_flow_no_chunks():
    """
    Tests the fallback 'not found' message.
    """
    with patch('src.retrieval.agent.QdrantRetriever') as mock_retriever:
        mock_retriever.return_value.retrieve_chunks.return_value = [] # No chunks found
        
        result = runner.invoke(app, ["An unanswerable question"])
        
        assert result.exit_code == 0
        assert "I could not find any information" in result.stdout
