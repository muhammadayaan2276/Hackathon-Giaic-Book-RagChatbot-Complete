from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk_text(self, extracted_data):
        chunks = []
        for item in extracted_data:
            doc_content = item["content"]
            doc_url = item["url"]
            doc_title = item["title"]

            splits = self.text_splitter.split_text(doc_content)
            
            for i, split in enumerate(splits):
                chunk_id = str(uuid.uuid4())
                chunk = {
                    "id": chunk_id,
                    "text": split,
                    "url": doc_url,
                    "title": doc_title,
                    "section": "", # This needs to be determined from the extractor or a more advanced chunking strategy
                    "chunk_index": i
                }
                chunks.append(chunk)
        return chunks
