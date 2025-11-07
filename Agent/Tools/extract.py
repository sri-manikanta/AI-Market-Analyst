# Extraction tool: retrieval + prompt + LLM call (Ollama)
from .ingest import Ingestor
from pydantic import BaseModel, Field, ValidationError, conlist
from langchain_ollama import ChatOllama

#Instantiate the model, specifying the model name
llm = ChatOllama(model="llama3.2")  # adjust to your local model name

class ExtractionSchema(BaseModel):
    company_name: str | None = Field(None, description="The official name of the company.")
    report_date: str | None = Field(None, description="The date the financial report was published (e.g., YYYY-MM-DD or Month Day, Year).")
    market_value_usd: float | None = Field(None, description="The total market valuation in US Dollars as a float (e.g., 12.34 or 1234000.0). Do not include dollar signs or commas.")
    cagr_percent: float | None = Field(None, description="The Compound Annual Growth Rate (CAGR) as a percentage float (e.g., 5.2 or 0.15). Do not include the '%' sign.")
    company_market_share_percent: float | None = Field(None, description="The company's market share as a percentage float. Do not include the '%' sign.")
    main_competitors: list[str] | None = Field(None, description="A list of the company's main competitors, each as a separate string.")
    SWOT: dict[str, list[str]] | None = Field(None, description="A dictionary containing 'Strengths', 'Weaknesses', 'Opportunities', and 'Threats' as keys, with lists of strings as values.")
    
structured_llm = llm.with_structured_output(schema=ExtractionSchema)

class ExtractTool:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

    def extract(self, prompt: str, top_k: int = 6) -> dict:
        chunks = self.ingestor.retrieve(query=prompt, top_k=top_k)
        ctx = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in chunks])
        system = (
            "You are a JSON extractor. Extract the requested fields and output ONLY valid JSON according to the schema. "
            "If a field is missing, output null for that field."
        )
        full_prompt = f"{system}\n\nContext:\n{ctx}\n\nInstructions:\n{prompt}\n\nOutput (only JSON):"
        response = structured_llm.invoke(full_prompt)
        print(f"[summ Tool] LLM Response: {response}")
        try:
            extraction = ExtractionSchema.model_validate(response)
            return extraction.model_dump()
        except ValidationError as ve:
            print(f"[Extract Tool] Validation Error: {ve}")
            return {"error": "Failed to parse extraction result."}
