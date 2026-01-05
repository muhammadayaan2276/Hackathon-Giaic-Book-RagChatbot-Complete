#!/usr/bin/env python3
"""
Migration script to update environment configuration for the FREE RAG agent.
"""

import os
import sys
from pathlib import Path


def migrate_env_file():
    """Migrate the .env file to the new configuration."""
    print("🔍 Checking for existing .env file...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ No .env file found. Creating a new one...")
        create_new_env_file()
        return
    
    print("✅ .env file found. Reading current configuration...")
    
    # Read the current .env file
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the current environment variables
    env_vars = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            if '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                env_vars[key] = value
    
    # Check for old variables
    old_vars = ['COHERE_API_KEY', 'OPENAI_API_KEY']
    has_old_vars = any(var in env_vars for var in old_vars)
    
    # Check for new variables
    new_vars = ['OPENROUTER_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
    has_new_vars = any(var in env_vars for var in new_vars)
    
    print(f"Old variables (to be removed): {old_vars if has_old_vars else 'None found'}")
    print(f"New variables (required): {new_vars if not has_new_vars else 'Already configured'}")
    
    # Create new .env content
    new_lines = []
    for line in lines:
        # Skip old variables
        if any(line.strip().startswith(old_var + '=') for old_var in old_vars):
            print(f"🗑️  Removing old variable: {line.strip()}")
            continue
        new_lines.append(line)
    
    # Add new required variables if not present
    new_required = []
    if 'OPENROUTER_API_KEY' not in env_vars:
        new_required.append('OPENROUTER_API_KEY=your-openrouter-api-key-here')
    if 'QDRANT_URL' not in env_vars:
        new_required.append('QDRANT_URL=your-qdrant-url-here')
    if 'QDRANT_API_KEY' not in env_vars:
        new_required.append('QDRANT_API_KEY=your-qdrant-api-key-here')
    
    if new_required:
        print(f"📝 Adding new required variables: {new_required}")
        # Add a blank line before new variables if needed
        if new_lines and new_lines[-1].strip() != '':
            new_lines.append('\n')
        new_lines.extend([var + '\n' for var in new_required])
    
    # Write the updated .env file
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ .env file updated successfully!")
    print("\n📋 Summary of changes:")
    print("- Removed COHERE_API_KEY and OPENAI_API_KEY (no longer needed)")
    print("- Added OPENROUTER_API_KEY (required for the free model)")
    print("- QDRANT_URL and QDRANT_API_KEY remain the same")
    print("\n💡 Remember to update your .env file with actual API keys!")


def create_new_env_file():
    """Create a new .env file with the required variables."""
    env_content = """# Environment variables for FREE RAG Agent
# Get your OpenRouter API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Qdrant configuration
QDRANT_URL=your-qdrant-url-here
QDRANT_API_KEY=your-qdrant-api-key-here
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ New .env file created with required variables.")
    print("\n📋 Remember to update your .env file with actual API keys!")


def install_dependencies():
    """Install the required dependencies."""
    print("\n📦 Installing required dependencies...")
    
    # Install sentence-transformers and other new dependencies
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "httpx"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 FREE RAG Agent Migration Tool")
    print("="*40)
    
    migrate_env_file()
    
    print("\n🔄 Installing dependencies...")
    install_dependencies()
    
    print("\n✅ Migration complete!")
    print("💡 You can now run the FREE RAG agent with:")
    print("   python src/retrieval/strict_rag_agent.py \"Your question here\"")