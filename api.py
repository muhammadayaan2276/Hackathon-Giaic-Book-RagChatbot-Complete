from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.retrieval.strict_rag_agent import StrictRAGAgent, LocalEmbedder, QdrantRetriever

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",  # Docusaurus default port
    "http://localhost:3002",  # Alternative Docusaurus port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    selected_text: str | None = None

class ChatResponse(BaseModel):
    answer: str
    source: str | None = None
    
@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}
 

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    embedder = LocalEmbedder()
    retriever = QdrantRetriever(embedder)
    agent = StrictRAGAgent()

    query = request.query
    if request.selected_text:
        query = f"Context: {request.selected_text}\n\nQuestion: {request.query}"

    retrieved_chunks = retriever.retrieve_chunks(query)
    answer = await agent.generate_answer(query, retrieved_chunks)

    return {"answer": answer}
