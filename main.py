import os

import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from Agent.Tools.ingest import Ingestor
from Agent.agent import MarketAgent as Agent
from google.adk.cli.fast_api import get_fast_api_app



# Get the directory where main.py is located
#AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Example session service URI (e.g., SQLite)
#SESSION_SERVICE_URI = "sqlite:///./sessions.db"
# Example allowed origins for CORS
#ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
# Set web=True if you intend to serve a web interface, False otherwise
#SERVE_WEB_INTERFACE = True

# Config from env
DATA_FILE = os.environ.get("DATA_FILE", "Agent/Data/innovate_inc_q3_2025.txt")
FAISS_DIR = os.environ.get("FAISS_DIR", "Agent/Vectors")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-large-v2")


# Call the function to get the FastAPI app instance
# Ensure the agent directory name ('capital_agent') matches your agent folder
app= FastAPI (title="AI Market Analyst Agent")
'''get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=SESSION_SERVICE_URI,
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
)'''

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ingestion (loads or creates FAISS index)
ingestor = Ingestor(data_file=DATA_FILE, faiss_dir=FAISS_DIR, embedding_model_name=EMBEDDING_MODEL)
ingestor.ensure_index()

# Initialize agent with tool bindings
agent = Agent(ingestor=ingestor)

@app.post("/agent")
async def agent_endpoint(payload: dict):
    """
    Conversational agent endpoint. Expects:
    {
      "query": "...",
      "history": [ {"role":"user","text":"..."}, ... ]  # Optional
    }
    """
    query = payload.get("query")
    history = payload.get("history", [])
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in payload")
    response = Agent.run_conversational(query=query, history=history)
    return {"response": response}

@app.post("/qa")
async def qa_endpoint(payload: dict):
    """
    QA endpoint: { "question": "...", "top_k": 4 }
    """
    question = payload.get("question")
    top_k = int(payload.get("top_k", 4))
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")
    ans = Agent.tools["qa"](question=question, top_k=top_k)
    return {"answer": ans}

@app.post("/summarize")
async def summarize_endpoint(payload: dict):
    """
    Summarize endpoint: { "prompt": "summarize market", "top_k": 6 }
    """
    prompt = payload.get("prompt", "Summarize the document")
    top_k = int(payload.get("top_k", 6))
    summary = Agent.tools["summarize"].summarize(prompt=prompt, top_k=top_k)
    return {"summary": summary}

@app.post("/extract")
async def extract_endpoint(payload: dict):
    """
    Extract endpoint: { "field_prompt": "...", "top_k": 6 }
    field_prompt: instructions to extract structured fields (e.g., JSON schema)
    """
    field_prompt = payload.get("field_prompt")
    top_k = int(payload.get("top_k", 6))
    if not field_prompt:
        raise HTTPException(status_code=400, detail="Missing field_prompt")
    res = Agent.tools["extract"].extract(field_prompt=field_prompt, top_k=top_k)
    return {"extraction": res}

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """
    Upload new external data file and embed it (append to FAISS).
    """
    contents = await file.read()
    filename = file.filename
    dest_path = os.path.join("data", filename)
    with open(dest_path, "wb") as f:
        f.write(contents)
    # Ingest and append
    ingestor.ingest_and_append_file(dest_path)
    return {"status": "ok", "message": f"Uploaded and ingested {filename}"}


if __name__ == "__main__":
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))