#!/usr/bin/env python3
"""
CLI entrypoint for the strict RAG agent.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from retrieval.strict_rag_agent import app


if __name__ == "__main__":
    # Pass command line arguments to the Typer app
    if len(sys.argv) < 2:
        print("Usage: python -m cli \"Your question here\"")
        print("Or: python cli.py \"Your question here\"")
        sys.exit(1)
    
    # Run the Typer app
    app()