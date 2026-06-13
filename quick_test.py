#!/usr/bin/env python3
"""
Quick test to verify basic functionality without complex imports.
"""

import os
import sys

# Check if required packages are installed
try:
    import sentence_transformers
    print("✅ sentence-transformers: available")
except ImportError:
    print("❌ sentence-transformers: not installed")

try:
    import requests
    print("✅ requests: available")
except ImportError:
    print("❌ requests: not installed")

# Check .env file
print(f"\nChecking .env file...")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print(f"QDRANT_URL: {qdrant_url}")
print(f"QDRANT_API_KEY: {'***' if qdrant_api_key else 'NOT SET'}")

if qdrant_url and qdrant_api_key:
    print("\n✅ QDRANT credentials found in .env")
    print("To test connection, you'll need to fix the pydantic version issue:")
    print("Run in command prompt:")
    print("pip install pydantic==2.5.3 pydantic-core==2.14.6 qdrant-client==1.8.0")
else:
    print("\n❌ Please set QDRANT_URL and QDRANT_API_KEY in .env file")

print("\nYour reingestion script is ready at:")
print("reingest_with_local_embeddings.py")