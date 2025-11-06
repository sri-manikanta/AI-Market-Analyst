# Tools/Summarize.py
from .ingest import Ingestor
import os
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama2")

class SummarizeTool:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

    def summarize(self, prompt: str = "Summarize the document", top_k: int = 6) -> str:
        # gather top_k chunks for the "document"
        chunks = self.ingestor.retrieve(prompt, top_k=top_k)
        ctx = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in chunks])
        system = (
            "You are an expert market analyst. Produce an executive summary (one paragraph) followed by 4 bullet points of key findings."
            " Base your summary strictly on the context."
        )
        full_prompt = f"{system}\n\nContext:\n{ctx}\n\nPrompt: {prompt}\n\nOutput:"
        payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "max_tokens": 400, "temperature": 0.2}
        url = f"{OLLAMA_HOST}/api/generate"
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "generated" in data:
                return data["generated"]
            if "choices" in data and data["choices"]:
                return data["choices"][0].get("message", {}).get("content", "")
            return str(data)
        except Exception as e:
            return f"[ERROR] Ollama call failed: {e}"
