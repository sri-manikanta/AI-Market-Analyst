# Question and Answer tool: retrieval + prompt + LLM call (Ollama)
from typing import List
from .ingest import Ingestor
from langchain_ollama import ChatOllama

#Instantiate the model, specifying the model name
llm = ChatOllama(model="llama3.2")  # adjust to your local model name

class QuestionAnswerTool:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

    def build_context(self, prompt: str, top_k: int = 4) -> str:
        chunks = self.ingestor.retrieve(query= prompt, top_k=top_k)
        ctx = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in chunks])
        return ctx

    def answer(self, prompt: str, top_k: int = 4) -> str:
        print(f"[QA Tool] Retrieving top {top_k} chunks for question: {prompt}")
        context = self.build_context(prompt= prompt, top_k= top_k)
        print (f"[QA Tool] Built context for question '{prompt}':\n{context}\n")
        system = (
            "You are a factual assistant. Answer the user's question strictly based on the provided context. "
            "If the answer is not present in the context, say 'INSUFFICIENT_DATA'. Do not hallucinate."
        )
        full_prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        response = llm.invoke(full_prompt)
        print(f"[QA Tool] LLM Response: {response.content}")
        return response.content