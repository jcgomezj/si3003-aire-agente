"""Cliente del LLM usando un endpoint compatible con OpenAI.

Funciona con Groq, NVIDIA NIM u Ollama cambiando solo las variables
de entorno (ver .env.example). Mantenemos la dependencia aislada aquí
para poder cambiar de proveedor sin tocar el loop del agente.
"""
from __future__ import annotations

from openai import OpenAI

from src import config


def get_client() -> OpenAI:
    """Crea un cliente OpenAI apuntando al proveedor configurado."""
    if not config.LLM_API_KEY:
        raise RuntimeError(
            "Falta LLM_API_KEY. Copia .env.example a .env y pon tu API key "
            "(ej. de https://console.groq.com/keys)."
        )
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def chat(messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2):
    """Una llamada de chat completions. Devuelve el objeto `message` de la respuesta.

    Args:
        messages: historial en formato OpenAI (role/content).
        tools: definiciones de herramientas (tool calling) opcionales.
        temperature: aleatoriedad del muestreo.
    """
    client = get_client()
    kwargs: dict = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message
