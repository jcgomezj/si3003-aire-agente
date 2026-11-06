"""Configuración central del proyecto.

Carga variables de entorno desde `.env` y expone rutas y parámetros
usados por el agente, el RAG, los datos y el dashboard.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Raíz del repositorio (dos niveles arriba de este archivo: src/config.py -> repo/)
ROOT = Path(__file__).resolve().parent.parent

# Carga .env si existe (no falla si no está)
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM (endpoint compatible con OpenAI)
# ---------------------------------------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Tope de iteraciones del loop agéntico (recomendación del enunciado: T_max)
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))

# ---------------------------------------------------------------------------
# Rutas de datos y corpus
# ---------------------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CORPUS_DIR = ROOT / "corpus"
CHROMA_DIR = ROOT / "data" / "chroma_db"

# Parámetros del RAG
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# Umbrales de categoría PM2.5 (µg/m³) — basados en el ICA/AQI colombiano.
# Fuente de referencia: ver corpus/normativa_pm25.md
PM25_BREAKPOINTS = [
    (0, 12, "Buena"),
    (12.1, 37, "Moderada"),
    (37.1, 55, "Dañina a grupos sensibles"),
    (55.1, 150, "Dañina a la salud"),
    (150.1, 250, "Muy dañina"),
    (250.1, 500, "Peligrosa"),
]


def pm25_category(value: float) -> str:
    """Devuelve la categoría de calidad del aire para un valor de PM2.5."""
    for low, high, label in PM25_BREAKPOINTS:
        if low <= value <= high:
            return label
    return "Peligrosa" if value > 500 else "Desconocida"
