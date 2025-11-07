from google.adk.agents import Agent
import logging
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field
from google.adk.tools import FunctionTool
import os

from .Tools.qa import QuestionAnswerTool
from .Tools.summarize import SummarizeTool
from .Tools.extract import ExtractTool
from .Tools.ingest import Ingestor # Your data ingestion class


# Configure logging for better visibility 
logging.basicConfig(level=logging.INFO)

# Config from env
DATA_FILE = os.environ.get("DATA_FILE", "Agent/Data/innovate_inc_q3_2025.txt")
FAISS_DIR = os.environ.get("FAISS_DIR", "Agent/Vectors")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-large-v2")

#llm2 = LiteLlm(model="ollama/qwen3:4b")  # adjust to your local model name 
llm = "gemini-2.5-flash" # configure API key in env var GOOGLE_API_KEY 

# Initialize ingestion (loads or creates FAISS index)
ingestor = Ingestor(data_file=DATA_FILE, faiss_dir=FAISS_DIR, embedding_model_name=EMBEDDING_MODEL)
ingestor.ensure_index()

# Instantiate core logic tools (these are NOT the ADK tool wrappers yet)
qa_logic = QuestionAnswerTool(ingestor)
summarize_logic = SummarizeTool(ingestor)
extract_logic = ExtractTool(ingestor)


#Define the ADK Tool Functions (wrappers around the core logic)
def tool_qa_fn(prompt: str, top_k: int) -> str:
    """Answers a question using the internal market document knowledge base."""
    return qa_logic.answer(prompt=prompt, top_k=top_k)

def tool_summarize_fn(prompt:str, top_k: int) -> str:
    """Generates a summary based on relevant market documents."""
    return summarize_logic.summarize(prompt=prompt, top_k=top_k)

def tool_extract_fn(prompt:str, top_k: int) -> str:
    """Extracts structured data from market documents based on a prompt."""
    # Return as a string or a JSON-formatted string, as the Agent expects a string result
    result = extract_logic.extract(prompt=prompt, top_k=top_k)
    return str(result)
# Create the ADK FunctionTool objects
qa_tool = FunctionTool(
    func=tool_qa_fn
)
summarize_tool = FunctionTool(
    func=tool_summarize_fn
)
extract_tool = FunctionTool(
    func=tool_extract_fn
)


market_analyst_agent = Agent(
    name="market_analyst_agent",
    model= llm,
    tools=[qa_tool, summarize_tool, extract_tool],
    instruction="""
            "You are the **AI Market Analyst agent**. Your sole function is to provide accurate, data-backed responses to market inquiries based **EXCLUSIVELY** on the information retrieved from your three tools.

            **STRICT TOOL USAGE RULES & PARAMETER MAPPING:**
            1.  **Mandatory Tool Use:** You **MUST** use one of the provided tools for **EVERY** request.
            2.  **Tool Routing and Parameter Generation Strategy:** The user's request must be translated into the exact input parameters for the chosen tool.

                * **A. For Factual Q&A (`qa_tool`):**
                    * **User Intent:** Seeking a specific, concise fact or number.
                    * **Parameter 1:** `prompt` (Type: Direct query, e.g., "What was the year-over-year growth rate for Q4?")
                    * **Parameter 2:** `top_k` (Value: Low, typically **2** or **3**, for high-precision retrieval of the answer.)
                    * **Required Call:** `qa_tool(prompt=..., top_k=...)`

                * **B. For Summarization (`summarize_tool`):**
                    * **User Intent:** Asking for an overview, summary, or main takeaways of a document/topic.
                    * **Parameter 1:** `prompt` (Type: Instruction to summarize, e.g., "Provide a summary of the 2024 Strategic Planning document.")
                    * **Parameter 2:** `top_k` (Value: Moderate, typically **5** or **6**, to ensure enough context for a comprehensive summary.)
                    * **Required Call:** `summarize_tool(prompt=..., top_k=...)`

                * **C. For Data Extraction (`extract_tool`):**
                    * **User Intent:** Requesting specific, structured data points, lists, or structured outputs.
                    * **Parameter 1:** `prompt` (Type: Specific extraction instruction, e.g., "Extract the names, titles, and department of all members of the Risk Committee.")
                    * **Parameter 2:** `top_k` (Value: Moderate, typically **4** or **5**, to gather all fragments of data needed for the structured output.)
                    * **Required Call:** `extract_tool(prompt=..., top_k=...)`

            3.  **Final Output Format:** After receiving the tool's output, formulate a polite and professional response. **You MUST cite the tool and the main parameter used** in the final answer (e.g., "The `qa_tool` with `question='...'` reported that..." or "The data extracted by `extract_tool` using `field_prompt='...'` is as follows...").
            """,
    description="""
        "Experienced market analyst agent. Its primary function is to query market reports and internal documents using its three specialized tools (`qa_tool`, `summarize_tool`, `extract_tool`). This agent is highly proficient at translating a user's request into the **exact required tool parameters** (`prompt` along with `top_k`). **Crucially, this agent is constrained to only provide answers based on the output of these tools** and must cite the tool used." 
        """,
    output_key="response"
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = market_analyst_agent
