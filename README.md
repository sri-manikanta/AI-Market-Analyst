# AI-Market-Analyst
This agent ingest a provided market research document with external data and perform three distinct tasks: general Q&amp;A, market research  findings, and structured data extraction.

# Project Overview — “AI Market Analyst Agent” (VAIA Agentic AI Residency)

This repository implements an Agentic AI Market Analyst system built using the Google ADK framework for orchestration, FastAPI for endpoints, FAISS for retrieval, and Gradio for the user interface.
The goal is to build a structured, multi-tool agent that analyzes business reports, answers analytical questions, summarizes findings, and extracts structured information — all strictly grounded in the data provided by the user.

# System Architecture

Core Components:

Google ADK Agent Framework – Handles orchestration and routing of tasks to specialized tools.

FastAPI Server – Provides multiple endpoints for modular interaction.

FAISS Vector Store – Performs semantic retrieval from the embedded data corpus.

Embedding Model: intfloat/e5-large-v2 – Generates document embeddings for retrieval.

Gradio UI – Enables intuitive interaction for both tool-specific and conversational use cases.

Dockerized Deployment – Ensures consistent environment and easy deployment on Google Cloud Run.

# File Structure
AI-Market-Analyst/
│
├── README.md
├── .gitignore
├── Gradio.py              # Gradio user interface
├── Main.py                # FastAPI server entry point
├── Dockerfile
├── Requirements.txt
│
├── AI-Market-Analyst-Agent/
│   ├── __init__.py
│   ├── Agent.py            # Google ADK agent orchestration logic
│   ├── Data/
│   ├── Vectors/
│   └── Tools/
│       ├── Qa.py           # Question-answering tool
│       ├── Summarize.py    # Summary generation tool
│       ├── Extract.py      # Structured extraction tool
│       └── Ingest.py       # Data ingestion, chunking, and FAISS index creation

# Functional Endpoints (FastAPI)
Endpoint	Function	Description
/agent	AI-Market-Analyst-Agent	Conversational interface orchestrated by ADK; dynamically invokes tools based on query intent.
/qa	Question & Answer	Retrieves context from FAISS and answers queries grounded in data.
/summarize	Summarization	Generates concise executive summaries from the uploaded data.
/extract	Data Extraction	Extracts structured fields (company, market size, CAGR, competitors, SWOT) as strict JSON.
/upload	External Data Source	Allows users to upload additional or custom data files to embed and append to the FAISS index.

# Initialization Workflow

On server startup:

Check for existing FAISS index for the default dataset.

If index exists → load it.

Else → load dataset, embed using intfloat/e5-large-v2, and store FAISS index locally or in mounted storage.

Initialize the AI-Market-Analyst-Agent using Google ADK with three registered tools:

QuestionAnswerTool (uses QA pipeline)

SummarizeTool (uses summarization pipeline)

ExtractTool (uses structured extraction pipeline)

Start the FastAPI server exposing the five endpoints.

Initialize Gradio UI for visual interaction and testing.

# Tools and Their Logic
1. Question & Answer (Qa.py)

Retrieves top-K relevant chunks via FAISS.

Constructs context and invokes Gemini/Gemma model through ADK.

Ensures factual grounding — does not hallucinate beyond provided data.

2. Summarize (Summarize.py)

Combines retrieved text into a compact executive summary (paragraph + bullet points).

Uses temperature=0.2 for balanced creativity and precision.

3. Extract (Extract.py)

Prompts LLM to produce strict JSON with pre-defined schema.

Schema enforced via Pydantic / JSON Schema validation.

If invalid JSON is produced → auto-correct loop triggers.

4. Ingest (Ingest.py)

Reads data file (default or user-uploaded).

Chunks text into 500–800 token segments with 20% overlap.

Generates embeddings with intfloat/e5-large-v2.

Builds FAISS index (vectors.index, metadata.json) for future reuse.

# Gradio UI

Two selectable windows:

Individual Tools Panel:

Access QA, Summarize, and Extract endpoints independently for testing and demonstration.

Conversational Agent Panel:

Full AI-Market-Analyst-Agent mode for multi-turn analysis and contextual discussion.

Settings:

Upload custom data source file.

Select model (Gemini/Gemma/E5).
