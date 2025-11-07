# AI Market Analyst Agent

A sophisticated FastAPI-based agentic system that performs intelligent Q&A, summarization, and structured data extraction from financial market documents. This system combines vector retrieval, semantic search, and LLM-powered reasoning to provide accurate, context-aware insights.

---

## Table of Contents

1. [Setup & Run Instructions](#setup--run-instructions)
2. [Design Decisions](#design-decisions)
3. [API Usage](#api-usage)
4. [Project Structure](#project-structure)
5. [Environment Variables](#environment-variables)

---

## Setup & Run Instructions

### Prerequisites

Ensure you have the following installed on your system:

- **Python 3.8+**
- **pip** (Python package manager)
- **Google API Key** (for Gemini models) - set as environment variable `GOOGLE_API_KEY`
- **Optional:** Ollama (if using local LLM models)

### Step 1: Clone & Navigate to Project

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- `google-adk` - Google Agents & Tools SDK
- `uvicorn` - ASGI server for FastAPI
- `fastapi` - Modern Python web framework
- `pydantic` - Data validation
- `langchain_ollama` - LLM integration
- `gradio` - Optional UI framework
- `requests` - HTTP client
- `sentence_transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `tqdm` - Progress bars

### Step 4: Prepare Data File

Place your financial market document(s) in the data directory:

```bash
mkdir -p Agent/Data
# Place your .txt file (e.g., innovate_inc_q3_2025.txt) in Agent/Data/
```

### Step 5: Set Environment Variables

Create a `.env` file in the project root:

```bash
# Core Configuration
DATA_FILE=Agent/Data/innovate_inc_q3_2025.txt
FAISS_DIR=Agent/Vectors
EMBEDDING_MODEL=intfloat/e5-large-v2
GOOGLE_API_KEY=your_google_api_key_here
API_BASE=http://localhost:8000
```

### Step 6: Initialize Vector Index

On first run, the system will automatically create the FAISS index. If you need to manually rebuild:

```python
from Agent.Tools.ingest import Ingestor

ingestor = Ingestor(
    data_file="Agent/Data/innovate_inc_q3_2025.txt",
    faiss_dir="Agent/Vectors",
    embedding_model_name="intfloat/e5-large-v2"
)
ingestor.ensure_index()
```

### Step 7: Run the FastAPI Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

### Step 8: (Optional) Run Gradio Interface

In a separate terminal:

```bash
python gradio_app.py
```

The Gradio interface will be available at `http://localhost:7860`

---

## Design Decisions

### 1. Chunking Strategy

**Configuration:** `chunk_size=50` tokens, `chunk_overlap=0.2` (20%)

**Rationale:**

- **Chunk Size (50 tokens):** Financial documents contain dense, interconnected information. A 50-token chunk (~150-200 characters) provides:
  - **Fine-grained retrieval:** Allows precise matching of specific data points (e.g., financial metrics, dates)
  - **Context preservation:** Maintains semantic coherence within each chunk
  - **Balance:** Avoids excessive fragmentation while keeping chunks manageable
  - **Token efficiency:** Reduces embedding costs when processing large documents

- **Overlap (20%):** Overlapping chunks solve the "boundary problem" where critical information straddles chunk boundaries:
  - **Semantic continuity:** A fact split between two chunks will appear in both, ensuring retrieval
  - **20% overlap** (10 tokens) provides sufficient redundancy without excessive duplication
  - **Retrieval robustness:** Query embeddings are more likely to match overlapping context

**Alternative Considered:** Larger chunks (100+ tokens) were rejected because financial Q&A requires precision; overly broad chunks dilute relevant information with noise.

---

### 2. Embedding Model

**Selected Model:** `intfloat/e5-large-v2`

**Rationale:**

- **Domain Performance:** E5 (Embeddings from Bidirectional Encoder Representations from Transformers) excels at:
  - **Financial terminology:** Understands domain-specific language (CAGR, market share, valuation, etc.)
  - **Semantic similarity:** Strong performance on retrieval-augmented generation (RAG) tasks
  - **Multi-lingual support:** Can handle international market documents

- **Model Specifications:**
  - **Dimension:** 1024-dimensional vectors (standard for modern embeddings)
  - **Architecture:** Bidirectional encoder (captures full context)
  - **Training:** Fine-tuned on 430M English sentence pairs; optimized for similarity tasks

- **Advantages over alternatives:**
  - vs. `text-embedding-3-small`: E5-large is cheaper (open-source) while maintaining comparable quality
  - vs. `all-MiniLM-L6-v2`: E5-large has superior performance on financial/technical domains
  - vs. `GIST-large-embedding-v0`: E5 has better community support and deployment flexibility

- **Trade-off:** Larger model (~435M parameters) requires more memory (~1.7 GB) but provides superior retrieval accuracy for financial documents.

---

### 3. Vector Database

**Selected Database:** FAISS (Facebook AI Similarity Search) - CPU version (`faiss-cpu`)

**Rationale:**

- **Why FAISS:**
  - **Exact KNN Search:** FAISS provides exact nearest-neighbor retrieval, ensuring no relevant documents are missed
  - **Speed:** Can search millions of vectors in milliseconds
  - **Local-first:** Operates entirely in-memory on CPU, eliminating network latency and cloud dependency
  - **JSON Metadata:** Paired with local JSON storage for full chunk text and source tracking
  - **Simplicity:** No external database setup required; perfect for development and deployment

- **Architecture:**
  - **Index Type:** Flat (exact search) with L2 distance metric
  - **Metadata Storage:** JSON file (`metadata.json`) stores chunk text and source information
  - **Persistence:** Index saved to disk; automatically loads on startup

- **Alternatives Considered:**
  - **Weaviate/Milvus:** Overkill for single-document use cases; adds operational complexity
  - **Pinecone:** Cloud-dependent; incurs per-request costs; less suitable for secure enterprise environments
  - **FAISS with GPU:** Chosen CPU version for portability; GPU variant available if performance becomes bottleneck

- **Scaling Path:** For multi-million document scenarios, FAISS can be replaced with:
  ```python
  index = faiss.IndexIVFFlat(quantizer, d, nlist)  # Hierarchical clustering
  ```

---

### 4. Data Extraction Prompt

**Objective:** Extract structured financial data (company name, market value, CAGR, market share) as validated JSON.

**Prompt Engineering Strategy:**

```python
system_prompt = """
You are a financial data extraction expert. Extract the requested information from the provided context 
and return it as a valid JSON object. If a field is not found, set its value to null.

STRICT RULES:
1. Return ONLY valid JSON (no markdown, no explanations)
2. Numeric fields must be raw numbers (no currency symbols, commas, or percentage signs)
3. Dates must be in YYYY-MM-DD format or exact text as found
4. Never hallucinate or infer missing data
"""
```

**Design Decisions:**

1. **Pydantic Schema Definition:**
   ```python
   class ExtractionSchema(BaseModel):
       company_name: str | None = Field(None, description="Official company name")
       market_value_usd: float | None = Field(None, description="Market valuation in USD (e.g., 1234000.0)")
       cagr_percent: float | None = Field(None, description="CAGR as percentage (e.g., 5.2)")
   ```
   - **Purpose:** Schema-based validation ensures LLM output conforms to expected structure
   - **Null handling:** Allows missing fields without breaking the parsing pipeline
   - **Field descriptions:** Guide the LLM to format numeric values correctly (no symbols, no commas)

2. **Reliability Techniques:**

   - **Constraint 1 - JSON-Only Output:**
     ```
     "Return ONLY valid JSON (no markdown, no explanations)"
     ```
     Prevents LLM from wrapping JSON in markdown code blocks, which would cause parsing failures.

   - **Constraint 2 - Numeric Format Specification:**
     ```
     "Numeric fields must be raw numbers (no currency symbols, commas, or percentage signs)"
     ```
     Without this, LLM might return `"1,234,567.89"` or `"5.2%"`, causing float parsing errors.

   - **Constraint 3 - Null Over Hallucination:**
     ```
     "Never hallucinate or infer missing data"
     ```
     Instructs LLM to return `null` rather than guess, maintaining data integrity.

   - **Constraint 4 - Date Standardization:**
     ```
     "Dates must be in YYYY-MM-DD format or exact text as found"
     ```
     Ensures consistent date parsing downstream.

3. **Prompt Anatomy:**
   ```
   System Prompt (constraints & format) 
   + Retrieved Context (top-k chunks)
   + User Query (specific extraction task)
   + Few-Shot Examples (optional, for complex extractions)
   ```

4. **Example Flow:**
   ```python
   user_query = "Extract company financials: name, market value, CAGR, and market share"
   
   full_prompt = f"""
   {system_prompt}
   
   Context:
   {retrieved_chunks_text}
   
   Task: {user_query}
   
   Return the extracted data as JSON:
   """
   
   response = llm.invoke(full_prompt)
   parsed_data = json.loads(response)  # Pydantic validates schema
   ```

5. **Error Handling:**
   ```python
   try:
       extraction_schema = ExtractionSchema.model_validate_json(llm_response)
   except ValidationError as e:
       # Fallback: retry with more specific prompt or return partial data
       pass
   ```

**Why This Works:**
- **Schema Validation:** Pydantic enforces type correctness before downstream processing
- **Explicit Constraints:** Removes ambiguity about numeric formatting and hallucination
- **Retrieval Context:** Limits LLM to document content, reducing fabrication
- **Null-Safe Design:** Gracefully handles missing fields without breaking pipelines

---

## API Usage

All endpoints accept POST requests with JSON payloads. The API exposes three specialized tools via FastAPI endpoints.

### Base URL

```
http://localhost:8000
```

---

### Task 1: Question Answering (`/tools/qa`)

Retrieve precise factual answers from the document knowledge base.

**Endpoint:**
```
POST /tools/qa
```

**Request Body:**
```json
{
  "prompt": "What was the year-over-year revenue growth in Q3 2025?",
  "top_k": 3
}
```

**Parameters:**
- `prompt` (string, required): The question to answer
- `top_k` (integer, optional, default=4): Number of document chunks to retrieve (typically 2-3 for precise answers)

**Response:**
```json
{
  "answer": "According to the Q3 2025 financial report, the year-over-year revenue growth was 12.5%, driven primarily by increased market penetration in North America and strong product demand."
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/tools/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What was the year-over-year revenue growth in Q3 2025?",
    "top_k": 3
  }'
```

**Example Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/tools/qa",
    json={
        "prompt": "What was the year-over-year revenue growth in Q3 2025?",
        "top_k": 3
    }
)

answer = response.json()["answer"]
print(answer)
```

**Example JavaScript:**
```javascript
fetch('http://localhost:8000/tools/qa', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "What was the year-over-year revenue growth in Q3 2025?",
    top_k: 3
  })
})
.then(res => res.json())
.then(data => console.log(data.answer))
```

---

### Task 2: Document Summarization (`/tools/summarize`)

Generate executive summaries with key findings from the knowledge base.

**Endpoint:**
```
POST /tools/summarize
```

**Request Body:**
```json
{
  "prompt": "Summarize the key achievements and challenges from Q3 2025",
  "top_k": 6
}
```

**Parameters:**
- `prompt` (string, optional): Summarization focus or context
- `top_k` (integer, optional, default=6): Number of chunks to retrieve (typically 5-6 for comprehensive summaries)

**Response:**
```json
{
  "summary": "Q3 2025 marked a pivotal period of growth and strategic repositioning for Innovate Inc. The company achieved record revenue of $450M, representing a 15% quarter-over-quarter increase, while successfully expanding operations into three new international markets. However, supply chain disruptions and increased competition in core segments present ongoing challenges.\n\nKey Findings:\n• Revenue Growth: 15% QoQ increase, reaching $450M\n• Market Expansion: Successfully entered markets in APAC and EMEA regions\n• Profitability: Operating margin improved to 22% from 19% in Q2\n• Headcount: Added 250 employees across R&D and sales functions"
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/tools/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize the key achievements and challenges from Q3 2025",
    "top_k": 6
  }'
```

**Example Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/tools/summarize",
    json={
        "prompt": "Summarize the key achievements and challenges from Q3 2025",
        "top_k": 6
    }
)

summary = response.json()["summary"]
print(summary)
```

**Example JavaScript:**
```javascript
fetch('http://localhost:8000/tools/summarize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "Summarize the key achievements and challenges from Q3 2025",
    top_k: 6
  })
})
.then(res => res.json())
.then(data => console.log(data.summary))
```

---

### Task 3: Structured Data Extraction (`/tools/extract`)

Extract structured, validated financial data as JSON from documents.

**Endpoint:**
```
POST /tools/extract
```

**Request Body:**
```json
{
  "prompt": "Extract company name, market valuation (USD), CAGR (percent), and market share (percent)",
  "top_k": 4
}
```

**Parameters:**
- `prompt` (string, required): Specific extraction instructions describing fields and formats
- `top_k` (integer, optional, default=4): Number of chunks to retrieve (typically 4-5 for structured extraction)

**Response:**
```json
{
  "extraction": {
    "company_name": "Innovate Inc.",
    "market_value_usd": 12500000000.0,
    "cagr_percent": 18.5,
    "company_market_share_percent": 8.2
  }
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/tools/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Extract company name, market valuation in USD, 5-year CAGR as a percentage, and market share as a percentage",
    "top_k": 4
  }'
```

**Example Python:**
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/tools/extract",
    json={
        "prompt": "Extract company name, market valuation in USD, 5-year CAGR as a percentage, and market share as a percentage",
        "top_k": 4
    }
)

extraction = response.json()["extraction"]
print(json.dumps(extraction, indent=2))
```

**Example JavaScript:**
```javascript
fetch('http://localhost:8000/tools/extract', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "Extract company name, market valuation in USD, 5-year CAGR as a percentage, and market share as a percentage",
    top_k: 4
  })
})
.then(res => res.json())
.then(data => console.log(JSON.stringify(data.extraction, null, 2)))
```

---

### Advanced: Agentic Interface

For sophisticated multi-turn conversations, use the agentic endpoint that automatically routes to appropriate tools:

**Endpoint:**
```
POST /run
```

**Request Body:**
```json
{
  "app_name": "Agent",
  "user_id": "user123",
  "session_id": "session_abc",
  "new_message": {
    "role": "user",
    "parts": [
      {
        "text": "Summarize Q3 2025 performance and extract the top 3 revenue drivers"
      }
    ]
  },
  "streaming": false
}
```

**Response:**
The agent automatically selects the appropriate tool(s) and returns synthesized results.

---

## Project Structure

```
.
├── Agent/
│   ├── __init__.py
│   ├── agent.py                    # Main agent orchestration
│   ├── Data/
│   │   └── innovate_inc_q3_2025.txt   # Source financial document
│   ├── Vectors/
│   │   ├── vectors.index           # FAISS index (binary)
│   │   └── metadata.json           # Chunk metadata
│   └── Tools/
│       ├── __init__.py
│       ├── ingest.py              # Document chunking & embedding
│       ├── qa.py                  # Question answering logic
│       ├── summarize.py           # Summarization logic
│       └── extract.py             # Structured extraction logic
├── main.py                         # FastAPI server definition
├── gradio_app.py                  # Optional Gradio UI
├── requirements.txt               # Python dependencies
└── README.md                       # This file
```

---

## Environment Variables

Create a `.env` file (or set system environment variables):

```bash
# Data Configuration
DATA_FILE=Agent/Data/innovate_inc_q3_2025.txt
FAISS_DIR=Agent/Vectors

# Embedding Model
EMBEDDING_MODEL=intfloat/e5-large-v2

# LLM Configuration
GOOGLE_API_KEY=your_google_api_key_here          # For Gemini models
# Alternatively, for local Ollama:
# OLLAMA_MODEL=llama3.2

# API Configuration
API_BASE=http://localhost:8000
SESSION_DB_URL=sqlite:///./sessions.db

# CORS Settings
ALLOWED_ORIGINS=http://localhost,http://localhost:8080,*
SERVE_WEB_INTERFACE=false
```

---

## Troubleshooting

### Issue: "FAISS index not found"
**Solution:** Run the index initialization:
```python
from Agent.Tools.ingest import Ingestor
ingestor = Ingestor()
ingestor.ensure_index()
```

### Issue: "GOOGLE_API_KEY not set"
**Solution:** Set your Google API key:
```bash
export GOOGLE_API_KEY="your_key_here"  # macOS/Linux
set GOOGLE_API_KEY=your_key_here       # Windows
```

### Issue: Slow embedding inference
**Solution:** Use GPU acceleration:
```bash
pip install faiss-gpu
# Ensure CUDA toolkit is installed
```

### Issue: Out of memory
**Solution:** Reduce `chunk_size` or use a smaller embedding model (e.g., `all-MiniLM-L6-v2`).

---

## Performance Considerations

- **Indexing Time:** ~2-5 minutes for a 50,000-word document
- **Query Latency:** 50-200ms (retrieval) + 1-5 seconds (LLM inference)
- **Memory Footprint:** ~2GB for embedding model + 500MB for typical FAISS index

---

## License & Attribution

This project leverages:
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
- [Google ADK](https://github.com/googleapis/python-adk) for agentic orchestration

---

## Support

For issues or questions, refer to the inline code documentation or open an issue in the repository.