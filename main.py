import os

import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from Agent.Tools.ingest import Ingestor
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
import google.adk.cli.fast_api as fast_api # Need this for internal paths
from pathlib import Path
from Agent.agent import tool_qa_fn, tool_summarize_fn, tool_extract_fn



# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Example session service URI (e.g., SQLite)
SESSION_DB_URL = "sqlite:///./sessions.db"
# Example allowed origins for CORS
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
# Set web=True if you intend to serve a web interface, False otherwise
SERVE_WEB_INTERFACE = False


tools_router = APIRouter(prefix="/tools", tags=["Custom tools"])


@tools_router.post("/qa")
async def qa_endpoint(payload: dict):
    """
    QA endpoint: { "prompt": "...", "top_k": 4 }
    """
    prompt= payload.get("prompt")
    top_k = int(payload.get("top_k", 4))
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing question")
    ans = tool_qa_fn(prompt=prompt, top_k=top_k)
    return {"answer": ans}

@tools_router.post("/summarize")
async def summarize_endpoint(payload: dict):
    """
    Summarize endpoint: { "prompt": "summarize market", "top_k": 6 }
    """
    prompt = payload.get("prompt", "Summarize the document")
    top_k = int(payload.get("top_k", 6))
    summary = tool_summarize_fn(prompt=prompt, top_k=top_k)
    return {"summary": summary}

@tools_router.post("/extract")
async def extract_endpoint(payload: dict):
    """
    Extract endpoint: { "prompt": "...", "top_k": 6 }
    field_prompt: instructions to extract structured fields (e.g., JSON schema)
    """
    prompt = payload.get("prompt")
    top_k = int(payload.get("top_k", 6))
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing field_prompt")
    res = tool_extract_fn(prompt=prompt, top_k=top_k)
    return {"extraction": res}

@tools_router.post("/upload")
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
    Ingestor.ingest_and_append_file(dest_path)
    return {"status": "ok", "message": f"Uploaded and ingested {filename}"}

# Call the function to get the FastAPI app instance
# Ensure the agent directory name ('capital_agent') matches your agent folder
app: FastAPI =  get_fast_api_app(
    agent_dir=AGENT_DIR,
    session_db_url=SESSION_DB_URL,
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
    trace_to_cloud=False,
)

# Include custom router
app.include_router(tools_router)

# Define paths needed for the ADK Web UI
BASE_DIR = Path(fast_api.__file__).parent.resolve()
ANGULAR_DIST_PATH = BASE_DIR / "browser"

# Define the necessary routes to redirect and serve index.html
@app.get("/")
async def redirect_to_dev_ui():
    """Redirects root to the main Dev UI page."""
    return RedirectResponse("/dev-ui")

@app.get("/dev-ui")
async def dev_ui():
    """Serves the index.html for the Web UI."""
    return FileResponse(ANGULAR_DIST_PATH / "index.html")

# Mount the static files at the root (the "catch-all" route)
app.mount("/", StaticFiles(directory=ANGULAR_DIST_PATH, html=True), name="static")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))