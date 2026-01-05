import pytest
import subprocess
import json
import os
from unittest.mock import patch

# Define the path to the cli.py script
CLI_PATH = "src/cli.py"

# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "COHERE_API_KEY": "z10Syb9KxYaCGkQrW1jmJdjX8I89sA5GbhfN662Y",
        "QDRANT_URL": "http://localhost:6333", # Use a local or mock Qdrant for integration tests
        "QDRANT_API_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.stzcRTS-RxJTFIaPu1t7lhuQACoLhdnNJm1ueaWNzis"
    }):
        yield

def run_cli_command(command_args):
    """Helper function to run cli.py commands."""
    result = subprocess.run(
        ["python", CLI_PATH] + command_args,
        capture_output=True,
        text=True,
        check=False # Do not raise an exception for non-zero exit codes
    )
    return result

def test_cli_retrieve_valid_query():
    """
    Integration test for 'retrieve' command with a valid query.
    Expects a non-empty output indicating retrieved chunks.
    Initially, this will likely show 'warning' or 'not implemented' messages.
    """
    query = "What is robotic AI?"
    top_k = 2
    result = run_cli_command(["retrieve", "--query", query, "--top-k", str(top_k)])
    
    # Expect success (exit code 0) but initially it might be non-zero
    # as the retrieval logic is not fully implemented yet.
    # We are checking for the presence of expected output patterns.
    assert result.returncode == 0 # Placeholder: will fail until retrieve logic is implemented
    assert "Retrieving top" in result.stdout
    assert "chunks" in result.stdout
    assert "CLI: Running retrieve command" in result.stdout
    # Further assertions will be added once retrieval is implemented to check content of chunks

def test_cli_retrieve_empty_query():
    """
    Integration test for 'retrieve' command with an empty query.
    Expects an error message from argparse about missing --query.
    """
    result = run_cli_command(["retrieve"])
    
    assert result.returncode != 0
    assert "the following arguments are required: --query" in result.stderr

def test_cli_retrieve_low_match_query():
    """
    Integration test for 'retrieve' command with a query expected to have low matches.
    Expects graceful handling, potentially empty results or specific log messages.
    """
    query = "A very unique and obscure phrase that won't match anything."
    top_k = 1
    result = run_cli_command(["retrieve", "--query", query, "--top-k", str(top_k)])
    
    assert result.returncode == 0 # Should complete gracefully
    assert "Retrieving top" in result.stdout
    assert "chunks" in result.stdout
    # Further assertions here will depend on the exact graceful handling implementation
    # For now, just ensuring it doesn't crash and processes the request.
    assert "No embedded chunks to upsert" not in result.stdout # Should not see this for retrieval
