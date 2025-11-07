# Summarization tool: retrieval + prompt + LLM call (Ollama)
from .ingest import Ingestor
import requests
from langchain_ollama import ChatOllama
from google.adk.tools import FunctionTool

#Instantiate the model, specifying the model name
llm = ChatOllama(model="llama3.2")  # adjust to your local model name


class SummarizeTool:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

    def summarize(self, prompt: str = "Summarize the document", top_k: int = 6) -> str:
        # gather top_k chunks for the "document"
        chunks = self.ingestor.retrieve(query=prompt, top_k=top_k)
        ctx = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in chunks])
        system = (
            "You are an expert market analyst. Produce an executive summary (one paragraph) followed by 4 bullet points of key findings."
            " Base your summary strictly on the context."
        )
        full_prompt = f"{system}\n\nContext:\n{ctx}\n\nPrompt: {prompt}\n\nOutput:"
        response = llm.invoke(full_prompt)
        print(f"[summ Tool] LLM Response: {response.content}")
        return response.content
