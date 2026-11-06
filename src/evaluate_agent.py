"""
Suite de Evaluación Experimental del Agente LLM + RAG (Entregable 4 - SI3003 Universidad EAFIT).

Ejecuta un banco fijo de 12 preguntas de prueba representativas de ciudadanos del Valle de Aburrá,
verificando cuantitativamente para cada pregunta:
1. Qué herramientas fueron invocadas por el loop agéntico (`get_pm25`, `predict_tomorrow`, `search_docs`).
2. Si el agente llamó a la(s) herramienta(s) esperada(s) según la intención de la pregunta.
3. Si la respuesta citó explícitamente la fuente normativa del RAG (`normativa_poeca_amva.md` / `DOC-`).
4. Si el manejo de errores como observación funcionó sin lanzar excepciones en preguntas con estaciones inválidas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from src.agent_rag import AirQualityAgent, AirQualityToolkit


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
        requiere_cita_rag=True,
    ),
    BenchmarkQuestion(
        id_pregunta="Q02",
        categoria_prueba="Predicción Parte 1",
        pregunta="¿Cuál es el pronóstico de PM2.5 para mañana en Itagüí según el modelo Naive Bayes y cuál es su probabilidad?",
        estacion_contexto="Casa de la Justicia (Itagüí)",
        herramientas_esperadas=["predict_tomorrow"],
        requiere_cita_rag=True,
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
        requiere_cita_rag=True,
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
        requiere_cita_rag=True,
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
        requiere_cita_rag=True,
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
    api_key: str = "",
    model_name: str = "llama-3.3-70b-versatile",
    t_max: int = 5,
) -> pd.DataFrame:
    """
    Ejecuta las 12 preguntas de prueba sobre el agente y construye una tabla de evaluación
    lista para incluir en el informe final y visualizar en el dashboard.
    """
    agent = AirQualityAgent(toolkit=toolkit, api_key=api_key, model_name=model_name, t_max=t_max)
    records = []

    for item in BENCHMARK_QUESTIONS:
        resp = agent.run(user_query=item.pregunta, default_station=item.estacion_contexto)
        called_tools = [step.tool_name for step in resp.steps]

        # Verificar si todas las herramientas esperadas fueron invocadas
        expected_called = all(exp_t in called_tools for exp_t in item.herramientas_esperadas)

        # Verificar si la respuesta cita explícitamente el corpus normativo
        ans_lower = resp.final_answer.lower()
        cited_source = ("normativa_poeca_amva.md" in ans_lower) or ("doc-" in ans_lower) or ("poeca" in ans_lower)

        # Verificar si capturó limpiamente una observación de error cuando correspondía
        handled_error_obs = any("OBSERVACIÓN DE ERROR" in step.observation for step in resp.steps)

        records.append({
            "ID": item.id_pregunta,
            "Categoría": item.categoria_prueba,
            "Pregunta de Prueba": item.pregunta,
            "Herramientas Esperadas": ", ".join(item.herramientas_esperadas),
            "Herramientas Invocadas": ", ".join(called_tools) if called_tools else "Ninguna",
            "¿Llamó Tool Correcta?": "Sí" if expected_called else "No",
            "¿Citó Fuente RAG?": "Sí" if cited_source else "No",
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

