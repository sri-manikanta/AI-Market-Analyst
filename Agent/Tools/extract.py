# Tools/Extract.py
from .ingest import Ingestor
import os
import requests
import json
from pydantic import BaseModel, ValidationError, Field

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama2")

class ExtractionSchema(BaseModel):
    company_name: str | None = None
    report_date: str | None = None
    market_value_usd: str | None = None
    cagr_percent: str | None = None
    company_market_share_percent: str | None = None
    main_competitors: list | None = None
    SWOT: dict | None = None

class ExtractTool:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

    def extract(self, field_prompt: str, top_k: int = 6) -> dict:
        chunks = self.ingestor.retrieve(field_prompt, top_k=top_k)
        ctx = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in chunks])
        system = (
            "You are a JSON extractor. Extract the requested fields and output ONLY valid JSON according to the schema. "
            "If a field is missing, output null for that field."
        )
        full_prompt = f"{system}\n\nContext:\n{ctx}\n\nInstructions:\n{field_prompt}\n\nOutput (only JSON):"
        payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "max_tokens": 800, "temperature": 0.0}
        url = f"{OLLAMA_HOST}/api/generate"
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            raw = ""
            if "generated" in data:
                raw = data["generated"]
            elif "choices" in data and data["choices"]:
                raw = data["choices"][0].get("message", {}).get("content", "")
            else:
                raw = json.dumps(data)
            # Try to parse JSON from raw text
            parsed = self._extract_json_from_text(raw)
            if parsed is None:
                return {"error": "Failed to parse JSON", "raw": raw}
            # Validate schema
            try:
                validated = ExtractionSchema(**parsed).dict()
                return validated
            except ValidationError as ve:
                return {"error": "Schema validation failed", "details": ve.errors(), "raw_parsed": parsed}
        except Exception as e:
            return {"error": f"Ollama call failed: {e}"}

    def _extract_json_from_text(self, text: str):
        # crude attempt: find first '{' and last '}' and parse
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                return None
            jtxt = text[start:end+1]
            return json.loads(jtxt)
        except Exception:
            # fallback: try to parse lines that look JSON-ish
            try:
                return json.loads(text)
            except Exception:
                return None
