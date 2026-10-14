# app package
import os

# transformers 5 carga pesos en paralelo con hilos; en Windows eso crashea (access violation)
# al cargar un segundo modelo (e5 + reranker). Carga secuencial: un poco más lenta, estable.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
