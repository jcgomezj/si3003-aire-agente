"""Loop agéntico MANUAL (ReAct con tool calling).

Flujo:
  1. Se envía el historial + definiciones de herramientas al LLM.
  2. Si el LLM pide una o más herramientas -> se ejecutan y se devuelven
     sus resultados como mensajes 'tool'.
  3. Se repite hasta que el LLM responde sin pedir herramientas, o hasta
     alcanzar AGENT_MAX_STEPS (T_max).

Devuelve la respuesta final y una traza de los pasos (útil para la
evaluación del agente y para mostrar en el dashboard).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src import config
from src.agent.llm_client import chat
from src.tools.tools import TOOL_SPECS, dispatch

SYSTEM_PROMPT = """Eres un asistente de calidad del aire para el Valle de Aburrá (Medellín y municipios cercanos).
Tu objetivo es ayudar a la ciudadanía a entender qué significa el PM2.5 del día y qué hacer al respecto.

Reglas:
- Usa las herramientas disponibles para obtener datos y fundamentar tus respuestas; no inventes cifras.
- Cuando des recomendaciones de salud o cites normativa/protocolos, PRIMERO consulta `search_docs` y CITA la fuente (nombre del documento).
- Para datos actuales usa `get_pm25`; para el día siguiente usa `predict_tomorrow`.
- Responde en español, de forma clara y breve.
- No eres un consejero médico: ante síntomas, recomienda consultar a un profesional de salud.
- Si una herramienta devuelve un error, explícalo con transparencia e intenta una alternativa.
"""


@dataclass
class AgentResult:
    answer: str
    steps: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


def run_agent(user_message: str, history: list[dict] | None = None) -> AgentResult:
    """Ejecuta el loop agéntico sobre un mensaje del usuario."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    steps: list[dict] = []
    tools_used: list[str] = []

    for _ in range(config.AGENT_MAX_STEPS):
        msg = chat(messages, tools=TOOL_SPECS)

        # ¿El modelo pidió herramientas?
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return AgentResult(
                answer=msg.content or "", steps=steps, tools_used=tools_used
            )

        # Registramos el turno del asistente (con las tool_calls) en el historial
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Ejecutamos cada herramienta y devolvemos la observación
        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            observation = dispatch(name, args)
            tools_used.append(name)
            steps.append({"tool": name, "args": args, "observation": observation})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )

    # Si agotamos T_max, pedimos una respuesta final sin herramientas
    messages.append(
        {
            "role": "user",
            "content": "Resume tu mejor respuesta con la información recabada.",
        }
    )
    final = chat(messages)
    return AgentResult(
        answer=final.content or "(sin respuesta)", steps=steps, tools_used=tools_used
    )


if __name__ == "__main__":
    result = run_agent("¿Cómo está el aire hoy en Medellín-Centro y qué debo hacer?")
    print("RESPUESTA:\n", result.answer)
    print("\nHERRAMIENTAS USADAS:", result.tools_used)
