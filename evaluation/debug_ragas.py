"""Debug script para identificar la incompatibilidad de RAGAS."""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from google import genai as google_genai
from ragas.llms.base import llm_factory, InstructorLLM, BaseRagasLLM
from ragas.metrics.collections import Faithfulness, ContextPrecision
from ragas.embeddings import GoogleEmbeddings

from app.config import get_settings
settings = get_settings()

print(f"Model: {settings.gemini_model}")

client = google_genai.Client(api_key=settings.google_api_key)
print(f"Client type: {type(client).__module__}.{type(client).__name__}")

llm = llm_factory(model=settings.gemini_model, provider="google", client=client)
print(f"LLM type: {type(llm).__module__}.{type(llm).__name__}")
print(f"Is InstructorLLM: {isinstance(llm, InstructorLLM)}")
print(f"Is BaseRagasLLM: {isinstance(llm, BaseRagasLLM)}")

try:
    m = Faithfulness(llm=llm)
    print(f"Faithfulness created OK: {type(m)}")
except Exception as e:
    print(f"Faithfulness FAILED: {type(e).__name__}: {e}")

try:
    m2 = ContextPrecision(llm=llm)
    print(f"ContextPrecision created OK: {type(m2)}")
except Exception as e:
    print(f"ContextPrecision FAILED: {type(e).__name__}: {e}")
