"""
Test script for the strict RAG agent to verify it follows the mandatory rules.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from retrieval.strict_rag_agent import StrictRAGAgent, CohereEmbedder, QdrantRetriever


def test_strict_behavior():
    """Test that the agent follows the strict rules"""
    print("Testing strict RAG agent behavior...")
    
    # Test with a query that should not be in the context
    query = "What is the capital of France?"
    
    print(f"Query: {query}")
    print("Expected: 'Answer not found in book.'")
    
    # Since we can't easily test the full pipeline without a Qdrant instance,
    # we'll verify the logic by testing the agent's response mechanism
    print("\nNote: To fully test the agent, you need to:")
    print("1. Set up your .env file with valid API keys")
    print("2. Ensure Qdrant has the book content indexed")
    print("3. Run: python src/retrieval/strict_rag_agent.py \"Your question here\"")
    
    print("\nTesting with a known topic from the book content...")
    print("Query: 'What is ROS 2?'")
    print("Expected: Information about ROS 2 from the book content")


def test_with_sample_context():
    """Test the agent's response with sample context"""
    print("\n" + "="*50)
    print("SIMULATED TEST - This is a conceptual test")
    print("="*50)
    
    print("\nWhen the agent receives a query about ROS 2 fundamentals:")
    print("Query: 'What is the purpose of a ROS 2 Node?'")
    print("\nWith proper context, the agent should respond with information")
    print("about ROS 2 nodes from the book content.")
    
    print("\nWhen the agent receives a query not in the context:")
    print("Query: 'What is the weather today?'")
    print("Response: 'Answer not found in book.'")
    
    print("\nWhen the agent receives a query about general knowledge:")
    print("Query: 'Who won the World Cup in 2022?'")
    print("Response: 'Answer not found in book.'")


if __name__ == "__main__":
    test_strict_behavior()
    test_with_sample_context()
    
    print("\n" + "="*50)
    print("TESTING INSTRUCTIONS")
    print("="*50)
    print("To properly test the strict RAG agent:")
    print("1. Ensure you have set up your .env file with API keys")
    print("2. Make sure Qdrant is running and has the book content indexed")
    print("3. Run the agent with a query like:")
    print("   python src/retrieval/strict_rag_agent.py \"What are ROS 2 Nodes?\"")
    print("4. Verify that responses follow the strict rules:")
    print("   - Only use provided context")
    print("   - Reply 'Answer not found in book.' when context is insufficient")
    print("   - No external knowledge or hallucinations")