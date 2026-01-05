I'm unable to run the pipeline because of a persistent `ModuleNotFoundError: No module named 'langchain.text_splitter'`.

My attempts to directly import `langchain.text_splitter` in a Python console also failed, confirming that this specific module is not accessible in the current Python environment. This contradicts your previous statement that dependencies were installed successfully.

Please ensure that `langchain` and its necessary sub-modules, especially `langchain.text_splitter`, are correctly installed. You might try the following commands again to ensure a complete and successful installation:

```bash
pip install --upgrade --force-reinstall langchain langchain-community langchain-core python-dotenv beautifulsoup4 cohere qdrant-client
```

It's possible that `langchain`'s internal structure or dependencies have changed, or the previous installation attempts were incomplete.

I cannot proceed with executing the pipeline until this dependency issue is resolved. Please let me know once you have verified the installation.