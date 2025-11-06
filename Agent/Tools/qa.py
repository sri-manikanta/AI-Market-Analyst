# Tools/Qa.py
# Question and Answer tool: retrieval + prompt + LLM call (Ollama)

from typing import List
from .ingest import Ingestor
import os
import requests
import json

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama2")  # adjust to your local model name

class QuestionAnswerTool:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

    def build_context(self, question: str, top_k: int = 4) -> str:
        chunks = self.ingestor.retrieve(question, top_k=top_k)
        ctx = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in chunks])
        return ctx

    def call_ollama(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        url = f"{OLLAMA_HOST}/api/generate"
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Ollama may stream or return structure; adapt based on server; here we expect 'generated' or 'choices'
            if "generated" in data:
                return data["generated"]
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0].get("message", {}).get("content", "")
            return json.dumps(data)
        except Exception as e:
            return f"[ERROR] Ollama call failed: {e}"

    def answer(self, question: str, top_k: int = 4) -> str:
        context = self.build_context(question, top_k=top_k)
        system = (
            "You are a factual assistant. Answer the user's question strictly based on the provided context. "
            "If the answer is not present in the context, say 'INSUFFICIENT_DATA'. Do not hallucinate."
        )
        prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        resp = self.call_ollama(prompt)
        return resp
