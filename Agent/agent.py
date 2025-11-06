import os
import asyncio
from typing import Dict, Any, List

# --- ADK Imports ---
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
#from google.adk.models.google_genai import GoogleGenAI
from pydantic import BaseModel, Field

# --- Custom Tool Imports (Assume they are in a local path) ---
# NOTE: Replace these with your actual tool/model imports if they're not ADK-ready
from .Tools.qa import QuestionAnswerTool
from .Tools.summarize import SummarizeTool
from .Tools.extract import ExtractTool
from .Tools.ingest import Ingestor # Your data ingestion class

# --- Configuration ---
# Use an environment variable for the Gemini API Key
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
ADK_MODEL = LiteLlm(model="ollama/llama3.2")

## --- Pydantic Schemas for Tools ---
# Define the structured input for each tool, which is an ADK requirement for FunctionTool

class QAToolInput(BaseModel):
    """Input for answering a question against the market knowledge base."""
    question: str = Field(description="The specific question to be answered from the documents.")
    top_k: int = Field(default=4, description="The number of top results to retrieve from the index.")

class SummarizeToolInput(BaseModel):
    """Input for summarizing relevant market documents."""
    prompt: str = Field(description="A prompt guiding the summary content, e.g., 'recent trends'.")
    top_k: int = Field(default=6, description="The number of top documents to summarize.")

class ExtractToolInput(BaseModel):
    """Input for structured extraction from market documents."""
    field_prompt: str = Field(description="A prompt describing the fields to extract, e.g., 'stock ticker and Q3 revenue'.")
    top_k: int = Field(default=6, description="The number of top documents to use for extraction.")


class MarketAgent:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

        # Instantiate core logic tools (these are NOT the ADK tool wrappers yet)
        self.qa_logic = QuestionAnswerTool(ingestor)
        self.summarize_logic = SummarizeTool(ingestor)
        self.extract_logic = ExtractTool(ingestor)

        # 1. Define the ADK Tool Functions (wrappers around the core logic)
        def tool_qa_fn(input: QAToolInput) -> str:
            """Answers a question using the internal market document knowledge base."""
            return self.qa_logic.answer(question=input.question, top_k=input.top_k)

        def tool_summarize_fn(input: SummarizeToolInput) -> str:
            """Generates a summary based on relevant market documents."""
            return self.summarize_logic.summarize(prompt=input.prompt, top_k=input.top_k)

        def tool_extract_fn(input: ExtractToolInput) -> str:
            """Extracts structured data from market documents based on a prompt."""
            # Return as a string or a JSON-formatted string, as the Agent expects a string result
            result = self.extract_logic.extract(field_prompt=input.field_prompt, top_k=input.top_k)
            return str(result)

        # 2. Create the ADK FunctionTool objects
        self.qa_tool = FunctionTool(
            func=tool_qa_fn
        )
        self.summarize_tool = FunctionTool(
            func=tool_summarize_fn
        )
        self.extract_tool = FunctionTool(
            func=tool_extract_fn
        )
        self.tools = {
            "summarize": self.summarize_logic,
            "qa": self.qa_logic,
            "extract": self.extract_logic,
        }

        # 3. Create the ADK Agent (using the standard Agent class)
        self.agent = Agent(
            name="ai_market_analyst",
            model=ADK_MODEL,
            instruction=(
                "You are the **AI Market Analyst agent**. Your role is to provide data-backed "
                "responses for market inquiries. You **MUST** use the provided tools to answer "
                "any questions about market reports or internal documents. Always cite which tool was used."
            ),
            # Pass the FunctionTool objects
            tools=[self.qa_tool, self.summarize_tool, self.extract_tool]
        )

        # 4. Initialize Runner & session service
        self.runner = Runner(app_name="market_analyst_app",
            agent=self.agent,
            session_service=InMemorySessionService())

    async def run_conversational_async(self, user_id: str, query: str) -> str:
        """
        Runs a query and returns the final response string.
        Uses ADK's modern async_stream_query method.
        """
        final_response = ""
        # The user_id is used to manage conversation state/memory
        async for event in self.runner.async_stream_query(
            agent=self.agent,
            user_id=user_id,
            message=query,
        ):
            if event.type == "final_answer":
                final_response = event.data.text
                break
            # You can add logic here to print tool_call/tool_result events for debugging
        return final_response