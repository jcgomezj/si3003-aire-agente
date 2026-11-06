"""Herramientas (tool calling) que el agente puede invocar.

Cumple el requisito de >= 2 herramientas. Incluimos tres:
  1. get_pm25        -> consulta datos reales/históricos (capa de datos)
  2. predict_tomorrow -> ejecuta el MODELO CLÁSICO de la Parte 1
  3. search_docs     -> recupera del corpus RAG (para citar fuentes)

Cada función devuelve un dict serializable. Los errores se devuelven como
observación (nunca se lanzan hacia el loop) siguiendo la recomendación del
enunciado: "devuelva los errores de las herramientas como observación".
"""
from __future__ import annotations

import json

from src.classic.predictor import predict_category
from src.data.loader import list_estaciones, load_air_quality
from src.rag.retriever import search


# ---------------------------------------------------------------------------
# Implementaciones
# ---------------------------------------------------------------------------
def get_pm25(estacion: str) -> dict:
    """Devuelve el último PM2.5 registrado y su categoría para una estación."""
    try:
        df = load_air_quality()
        sub = df[df["estacion"] == estacion].sort_values("fecha")
        if sub.empty:
            return {
                "error": f"Estación '{estacion}' no encontrada.",
                "estaciones_disponibles": list_estaciones(),
            }
        last = sub.iloc[-1]
        return {
            "estacion": estacion,
            "fecha": str(last["fecha"].date()),
            "pm25": float(last["pm25"]),
            "categoria": last["categoria"],
        }
    except Exception as exc:  # noqa: BLE001 - devolvemos error como observación
        return {"error": f"Fallo en get_pm25: {exc}"}


def predict_tomorrow(estacion: str) -> dict:
    """Predice la categoría de calidad del aire del día siguiente (modelo clásico)."""
    try:
        return predict_category(estacion)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Fallo en predict_tomorrow: {exc}"}


def search_docs(query: str) -> dict:
    """Busca en el corpus (normativa, protocolos, salud) y devuelve fragmentos citables."""
    try:
        resultados = search(query)
        return {"query": query, "resultados": resultados}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Fallo en search_docs: {exc}"}


# ---------------------------------------------------------------------------
# Registro: nombre -> función
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "get_pm25": get_pm25,
    "predict_tomorrow": predict_tomorrow,
    "search_docs": search_docs,
}


def dispatch(name: str, arguments: str | dict) -> dict:
    """Ejecuta una herramienta por nombre con argumentos (str JSON o dict)."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Herramienta desconocida: {name}"}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"error": f"Argumentos inválidos para {name}: {arguments!r}"}
    try:
        return TOOL_FUNCTIONS[name](**arguments)
    except TypeError as exc:
        return {"error": f"Argumentos incorrectos para {name}: {exc}"}


# ---------------------------------------------------------------------------
# Definiciones JSON-schema (formato OpenAI tool calling)
# ---------------------------------------------------------------------------
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_pm25",
            "description": (
                "Consulta el último valor de PM2.5 y su categoría de calidad del "
                "aire para una estación de monitoreo del Valle de Aburrá."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {
                        "type": "string",
                        "description": "Nombre de la estación, p. ej. 'Medellín-Centro'.",
                    }
                },
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_tomorrow",
            "description": (
                "Predice la categoría de calidad del aire (PM2.5) del día siguiente "
                "para una estación, usando el modelo clásico de la Parte 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {
                        "type": "string",
                        "description": "Nombre de la estación.",
                    }
                },
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Busca en el corpus documental (normativa de PM2.5, protocolos de "
                "episodios críticos y recomendaciones de salud) para fundamentar y "
                "citar las respuestas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pregunta o tema a buscar en los documentos.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]
