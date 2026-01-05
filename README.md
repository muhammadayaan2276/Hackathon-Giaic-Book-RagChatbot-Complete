# hackathon-book-writing-ai-robotics
“A Docusaurus-powered hackathon book for learning AI-driven humanoid robotics, including ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems.”

## FREE RAG Agent CLI

This project includes a FREE Retrieval-Augmented Generation (RAG) agent that can answer questions about the book's content.

### How to Run

1.  **Prerequisites**: Ensure you have Python 3.9+ and have installed the dependencies from `requirements.txt`.
2.  **Configuration**: Copy the `.env.example` file to `.env` and fill in your API keys for OpenRouter and your Qdrant instance details.
3.  **Run**: Execute the agent from your terminal.

    ```bash
    python src/retrieval/strict_rag_agent.py "Your question about the book here"
    ```

### Features
- **FREE**: No cost for API usage (using OpenRouter's free model)
- **Local Embeddings**: Uses local sentence transformer model (no API required for embeddings)
- **Strict RAG**: Follows strict rules - answers only from provided context
- **No External Knowledge**: Does not use any external knowledge or hallucinate

For more detailed instructions, please see the quickstart guide at `specs/005-rag-agent-qdrant/quickstart.md`.
