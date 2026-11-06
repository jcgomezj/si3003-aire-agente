"""Ejecuta las preguntas de prueba fijas contra el agente y reporta resultados.

Ejecutar con:
    python -m eval.run_eval

Verifica, para cada pregunta: qué herramientas usó el agente y si citó la
fuente esperada. Imprime una tabla resumen. Requiere LLM_API_KEY configurada.
"""
from __future__ import annotations

from src.agent.agent import run_agent

# (pregunta, herramientas_esperadas, fuente_esperada_o_None)
CASOS = [
    ("¿Cómo está el aire hoy en Medellín-Centro?", {"get_pm25"}, None),
    ("¿Qué categoría de PM2.5 habrá mañana en Itagüí?", {"predict_tomorrow"}, None),
    (
        "¿Qué significa que el PM2.5 sea 'dañino a grupos sensibles'?",
        {"search_docs"},
        "normativa_pm25.md",
    ),
    (
        "¿Qué debo hacer si el aire está dañino para la salud?",
        {"search_docs"},
        "recomendaciones_salud.md",
    ),
    (
        "¿Qué es un estado de alerta en el protocolo de episodios?",
        {"search_docs"},
        "protocolo_episodios.md",
    ),
    ("¿Cuál es el PM2.5 en la estación 'Marte'?", {"get_pm25"}, None),
    (
        "¿Quiénes son los grupos sensibles a la contaminación?",
        {"search_docs"},
        "recomendaciones_salud.md",
    ),
]


def _cito_fuente(result, fuente: str | None) -> bool:
    if fuente is None:
        return True  # no aplica
    # ¿aparece la fuente en la respuesta o en las observaciones de search_docs?
    if fuente.split(".")[0] in result.answer:
        return True
    for step in result.steps:
        if step["tool"] == "search_docs":
            obs = step["observation"]
            for r in obs.get("resultados", []):
                if r.get("fuente") == fuente:
                    return True
    return False


def main() -> None:
    aciertos_tool = 0
    aciertos_cita = 0
    total_cita = 0

    print(f"{'#':>2} | {'tool_ok':7} | {'cita_ok':7} | pregunta")
    print("-" * 70)
    for i, (pregunta, tools_esp, fuente) in enumerate(CASOS, 1):
        result = run_agent(pregunta)
        tool_ok = bool(tools_esp & set(result.tools_used))
        cita_ok = _cito_fuente(result, fuente)
        aciertos_tool += int(tool_ok)
        if fuente is not None:
            total_cita += 1
            aciertos_cita += int(cita_ok)
        print(
            f"{i:>2} | {str(tool_ok):7} | {str(cita_ok):7} | {pregunta[:45]}"
        )

    print("-" * 70)
    print(f"Herramienta correcta: {aciertos_tool}/{len(CASOS)}")
    if total_cita:
        print(f"Citó fuente esperada: {aciertos_cita}/{total_cita}")


if __name__ == "__main__":
    main()
