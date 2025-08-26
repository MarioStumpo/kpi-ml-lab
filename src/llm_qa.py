# src/llm_qa.py
from __future__ import annotations
import os

try:
    import ollama
except Exception:
    ollama = None

SYSTEM_PROMPT = (
    "You are a precise sports performance analyst. "
    "Use ONLY the provided context (reports + stats). "
    "If something is not present, say you don't know. "
    "Prefer bullet points and cite metric names explicitly."
)

def call_local_llm(context: str, question: str) -> str:
    if ollama is None:
        raise RuntimeError("Ollama client not installed")
    model = os.getenv("OLLAMA_MODEL", "mistral:7b")
    try:
        resp = ollama.chat(model=model, messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"}
        ], options={"temperature":0.2}, stream=False)
        return resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")