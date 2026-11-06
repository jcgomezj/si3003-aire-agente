"""
Suite de Evaluación Experimental del Agente LLM + RAG (Entregable 4 - SI3003 Universidad EAFIT).

Ejecuta un banco fijo de 12 preguntas de prueba representativas de ciudadanos del Valle de Aburrá,
verificando con rigor experimental para cada pregunta:
1. Modo de ejecución real: usa la `GROQ_API_KEY` del entorno (`os.environ`) para evaluar el loop
   agéntico con el LLM real en Groq. Si no hay API key o ocurre fallback, lo reporta explícitamente
   en la columna `Modo de Ejecución` como `"MODO FALLBACK DETERMINISTA (no representa el agente LLM)"`.
2. Criterio estricto de acierto de herramientas (Igualdad de conjuntos `set(invocadas) == set(esperadas)`):
   solo marca `"Sí"` cuando el conjunto de herramientas llamadas coincide exactamente con el conjunto
   esperado (sin herramientas faltantes ni herramientas sobrantes).
3. Columna explícita `Tools Faltantes / Sobrantes` para auditar cualquier desviación del agente.
4. Verificación de citación de fuente RAG (`normativa_poeca_amva.md` / `DOC-` / `POECA`) y captura
   controlada de errores como observación (`Q11`).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.agent_rag import AirQualityAgent, AirQualityToolkit, _load_env_file_if_present


@dataclass
class BenchmarkQuestion:
    id_pregunta: str
    categoria_prueba: str
    pregunta: str
    estacion_contexto: str
    herramientas_esperadas: List[str]
    requiere_cita_rag: bool


BENCHMARK_QUESTIONS: List[BenchmarkQuestion] = [
    BenchmarkQuestion(
        id_pregunta="Q01",
        categoria_prueba="Consulta actual SIATA",
        pregunta="¿Cuál es el nivel actual de PM2.5 y la categoría del aire hoy en la estación de Bello?",
        estacion_contexto="U. San Buenaventura (Bello)",
        herramientas_esperadas=["get_pm25"],
        requiere_cita_rag=False,
    ),
    BenchmarkQuestion(
        id_pregunta="Q02",
        categoria_prueba="Predicción Parte 1",
        pregunta="¿Cuál es el pronóstico de PM2.5 para mañana en Itagüí según el modelo Naive Bayes y cuál es su probabilidad?",
        estacion_contexto="Casa de la Justicia (Itagüí)",
        herramientas_esperadas=["predict_tomorrow"],
        requiere_cita_rag=False,
    ),
    BenchmarkQuestion(
        id_pregunta="Q03",
        categoria_prueba="Normativa POECA / RAG",
        pregunta="Según la Resolución 2254 de 2017 y el POECA, ¿qué significa el nivel Naranja (Dañina a Grupos Sensibles) y qué rango de PM2.5 tiene?",
        estacion_contexto="Museo de Antioquia (Centro)",
        herramientas_esperadas=["search_docs"],
        requiere_cita_rag=True,
    ),
    BenchmarkQuestion(
        id_pregunta="Q04",
        categoria_prueba="Multiherramienta (Actual + Predicción + Salud)",
        pregunta="Vivo en el Centro de Medellín y tengo asma. ¿Cómo está el PM2.5 hoy, qué se espera para mañana y qué recomendaciones de salud debo seguir?",
        estacion_contexto="Museo de Antioquia (Centro)",
        herramientas_esperadas=["get_pm25", "predict_tomorrow", "search_docs"],
        requiere_cita_rag=True,
    ),
    BenchmarkQuestion(
        id_pregunta="Q05",
        categoria_prueba="Deporte al aire libre",
        pregunta="¿Cómo está el aire hoy en El Poblado y qué recomienda la normativa sobre salir a trotar al aire libre?",
        estacion_contexto="Tanques EPM (El Poblado)",
        herramientas_esperadas=["get_pm25", "search_docs"],
        requiere_cita_rag=True,
    ),
    BenchmarkQuestion(
        id_pregunta="Q06",
        categoria_prueba="Estado Latente HMM (Parte 1)",
        pregunta="¿Cuál es el estado latente del episodio ambiental (HMM) y la predicción de mañana para la estación de Envigado?",
        estacion_contexto="Hosp. Manuel Uribe Ángel (Envigado)",
        herramientas_esperadas=["predict_tomorrow"],
        requiere_cita_rag=False,
    ),
    BenchmarkQuestion(
        id_pregunta="Q07",
        categoria_prueba="Medidas Institucionales POECA",
        pregunta="¿Qué medidas institucionales como pico y placa ambiental se toman cuando el Valle de Aburrá entra en Estado de Alerta Roja según el POECA?",
        estacion_contexto="Museo de Antioquia (Centro)",
        herramientas_esperadas=["search_docs"],
        requiere_cita_rag=True,
    ),
    BenchmarkQuestion(
        id_pregunta="Q08",
        categoria_prueba="Consulta actual + Meteorología",
        pregunta="¿Cuál es la medición actual de PM2.5, lluvia y viento en la estación de Caldas?",
        estacion_contexto="E. Joaquín Aristizábal (Caldas)",
        herramientas_esperadas=["get_pm25"],
        requiere_cita_rag=False,
    ),
    BenchmarkQuestion(
        id_pregunta="Q09",
        categoria_prueba="Factores meteorológicos y topografía (RAG)",
        pregunta="¿Por qué la lluvia nocturna, el viento y la inversión térmica afectan el PM2.5 en el Valle de Aburrá según los documentos del SIATA?",
        estacion_contexto="Museo de Antioquia (Centro)",
        herramientas_esperadas=["search_docs"],
        requiere_cita_rag=True,
    ),
    BenchmarkQuestion(
        id_pregunta="Q10",
        categoria_prueba="Grupos Sensibles y Niños",
        pregunta="¿Qué grupos de personas se consideran grupos sensibles según el protocolo POECA y qué cuidados deben tener en alerta naranja?",
        estacion_contexto="Casa de la Justicia (Itagüí)",
        herramientas_esperadas=["search_docs"],
        requiere_cita_rag=True,
    ),
    BenchmarkQuestion(
        id_pregunta="Q11",
        categoria_prueba="Resiliencia / Manejo de Errores",
        pregunta="¿Cuál es el nivel actual de PM2.5 hoy en la estación de Bogotá Norte?",
        estacion_contexto="Museo de Antioquia (Centro)",
        herramientas_esperadas=["get_pm25"],
        requiere_cita_rag=False,
    ),
    BenchmarkQuestion(
        id_pregunta="Q12",
        categoria_prueba="Síntesis Completa Sur del Valle",
        pregunta="Dame el estado actual de PM2.5 en Caldas, el pronóstico de mañana y las recomendaciones del POECA.",
        estacion_contexto="E. Joaquín Aristizábal (Caldas)",
        herramientas_esperadas=["get_pm25", "predict_tomorrow", "search_docs"],
        requiere_cita_rag=True,
    ),
]


def run_agent_evaluation(
    toolkit: AirQualityToolkit,
    api_key: Optional[str] = None,
    model_name: str = "openai/gpt-oss-120b",
    t_max: int = 5,
) -> pd.DataFrame:
    """
    Ejecuta las 12 preguntas de prueba sobre el agente usando la `GROQ_API_KEY` del entorno
    cuando esté disponible, y evalúa con igualdad estricta de conjuntos las herramientas invocadas.
    """
    _load_env_file_if_present()
    effective_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "").strip()
    effective_model = os.environ.get("GROQ_MODEL", model_name)

    agent = AirQualityAgent(
        toolkit=toolkit,
        api_key=effective_key,
        model_name=effective_model,
        t_max=t_max,
    )
    records = []

    for item in BENCHMARK_QUESTIONS:
        resp = agent.run(user_query=item.pregunta, default_station=item.estacion_contexto)
        called_tools = [step.tool_name for step in resp.steps]

        expected_set = set(item.herramientas_esperadas)
        called_set = set(called_tools)

        # Criterio estricto: IGUALDAD de conjuntos entre herramientas esperadas e invocadas
        strict_match = called_set == expected_set

        missing_tools = sorted(list(expected_set - called_set))
        extra_tools = sorted(list(called_set - expected_set))
        diff_parts = []
        if missing_tools:
            diff_parts.append(f"Faltantes: {', '.join(missing_tools)}")
        if extra_tools:
            diff_parts.append(f"Sobrantes: {', '.join(extra_tools)}")
        diff_str = "Ninguna (Exacto)" if not diff_parts else " | ".join(diff_parts)

        # Verificar citación de fuente normativa cuando se invocó search_docs o en la respuesta
        ans_lower = resp.final_answer.lower()
        cited_source = (
            ("normativa_poeca_amva.md" in ans_lower)
            or ("doc-" in ans_lower)
            or ("poeca" in ans_lower)
            or ("2254" in ans_lower)
        )

        handled_error_obs = any("OBSERVACIÓN DE ERROR" in step.observation for step in resp.steps)

        if resp.used_fallback_mode or not effective_key:
            exec_mode = "MODO FALLBACK DETERMINISTA (no representa el agente LLM)"
        else:
            exec_mode = f"AGENTE LLM REAL (Groq: {effective_model})"

        records.append({
            "ID": item.id_pregunta,
            "Categoría": item.categoria_prueba,
            "Modo de Ejecución": exec_mode,
            "Pregunta de Prueba": item.pregunta,
            "Herramientas Esperadas": ", ".join(item.herramientas_esperadas),
            "Herramientas Invocadas": ", ".join(called_tools) if called_tools else "Ninguna",
            "¿Llamó Tool Correcta?": "Sí" if strict_match else "No",
            "Tools Faltantes / Sobrantes": diff_str,
            "¿Citó Fuente RAG?": "Sí" if cited_source else ("No (No requería RAG)" if not item.requiere_cita_rag else "No"),
            "Iteraciones Usadas": f"{resp.iterations_used}/{t_max}",
            "Observación de Error Capturada": "Sí (Controlado)" if handled_error_obs else "No aplica",
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    tk = AirQualityToolkit(
        data_csv_path=project_root / "data" / "siata_pm25_historico.csv",
        docs_rag_dir=project_root / "docs_rag",
    )
    df_results = run_agent_evaluation(tk)
    print("=== RESULTADOS DE EVALUACIÓN DEL AGENTE (12 PREGUNTAS DE PRUEBA) ===")
    print(df_results.to_string(index=False))
    out_csv = project_root / "data" / "evaluacion_agente_12_preguntas.csv"
    df_results.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nArchivo exportado exitosamente en: {out_csv}")
