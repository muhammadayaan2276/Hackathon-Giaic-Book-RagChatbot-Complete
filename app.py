import os
import httpx
import logging
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS Fix
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG Logic
class ChatbotEngine:
    def __init__(self):
        # Prefer local model path if exists
        local_model_path = "./models/all-MiniLM-L6-v2"
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            self.model = SentenceTransformer(local_model_path)
        else:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=120
        )
        self.collection = "docusaurus_book"
        
        # Load FAQ list
        self.faq = self.load_faq("docs/500_qa_list.md")

    def load_faq(self, file_path):
        faq = {}
        if not os.path.exists(file_path):
            file_path = "hackathon-Giaic/" + file_path
            
        if os.path.exists(file_path):
            import re
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            blocks = re.split(r'Q\d+:', content)
            for block in blocks:
                if 'A:' in block:
                    q, a = block.split('A:', 1)
                    faq[q.strip().lower()] = a.strip()
        return faq

    def retrieve(self, query: str):
        # First check FAQ
        faq_answer = self.faq.get(query.strip().lower())
        if faq_answer:
            return {"type": "faq", "text": faq_answer}
            
        try:
            vector = self.model.encode(query).tolist()
            results = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=5
            ).points
            return {"type": "rag", "text": "\n\n".join([r.payload.get("text", "") for r in results])}
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return {"type": "rag", "text": ""}

    async def generate(self, query: str, context_obj: Dict[str, str]):
        # If it was an FAQ match, return immediately
        if context_obj["type"] == "faq":
            return context_obj["text"]
            
        context = context_obj["text"]
        # ... rest of the generate method ...
        # (Using the previously established API logic)
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "https://huggingface.co/spaces",
            "X-Title": "Robotics-Book-RAG"
        }
        
        prompt = f"""You are a helpful Robotics assistant. Use the provided context to answer the user's question accurately.
Context:
{context}

Question: {query}

If the answer is present in the context, provide a detailed answer. If the answer is not clearly found in the context, say 'Answer not found in book'."""
        
        payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, headers=headers, json=payload, timeout=30.0)
                if res.status_code == 200:
                    return res.json()['choices'][0]['message']['content']
                return f"Error: API returned {res.status_code}. Answer not found in book."
        except Exception as e:
            return "Sorry, I encountered an error. Please try again later."

engine = ChatbotEngine()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def home(): 
    return {"status": "Active", "message": "Robotics RAG API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    context = engine.retrieve(request.query)
    answer = await engine.generate(request.query, context)
    return {"answer": answer}
