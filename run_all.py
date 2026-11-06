"""
Orquestador Principal (`run_all.py`) - Proyecto Final SI3003 Inteligencia Artificial (EAFIT).

Ejecuta el flujo completo de principio a fin con un solo comando:
1. Carga, limpia y preprocesa los datos históricos del SIATA (`src/data_loader.py`).
2. Ejecuta el Análisis Exploratorio de Datos (EDA) y exporta tablas/visualizaciones (`src/eda.py`).
3. Entrena el modelo clásico de la Parte 1, calcula métricas vs. líneas base y guarda el
   artefacto serializado en `models/air_quality_model_part1.pkl` (`src/model_part1.py`).
4. Ejecuta la evaluación experimental de las 12 preguntas de prueba del Agente (`src/evaluate_agent.py`).
5. Lanza automáticamente el Dashboard interactivo en Streamlit (`src/app.py`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.agent_rag import AirQualityToolkit
from src.data_loader import load_air_quality_data
from src.eda import run_eda_pipeline
from src.evaluate_agent import run_agent_evaluation
from src.model_part1 import ClassicalAirQualityModel, DEFAULT_MODEL_PATH


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print("=" * 75)
    print("INICIANDO PIPELINE COMPLETO - SISTEMA DE CALIDAD DEL AIRE (EAFIT SI3003)")
    print("=" * 75)

    # Paso 1: Ingeniería de Datos y Plan B
    print("\n[1/5] Cargando y preprocesando datos históricos del SIATA (`src/data_loader.py`)...")
    df = load_air_quality_data()
    print(f"      [OK] Dataset listo: {df.shape[0]} registros x {df.shape[1]} columnas.")

    # Paso 2: Análisis Exploratorio de Datos (EDA)
    print("\n[2/5] Generando Análisis Exploratorio (EDA) y visualizaciones (`src/eda.py`)...")
    eda_res = run_eda_pipeline()
    print(f"      [OK] Tablas y gráficos exportados en `data/eda_outputs/` ({len(eda_res['summary_table'])} estaciones).")

    # Paso 3: Entrenamiento, Evaluación vs. Baselines y Serialización del Modelo Parte 1
    print("\n[3/5] Entrenando Modelo Clásico Parte 1 y guardando en `models/` (`src/model_part1.py`)...")
    model = ClassicalAirQualityModel()
    metrics = model.fit_and_evaluate(df)
    saved_model = model.save_model(DEFAULT_MODEL_PATH)
    print(f"      [OK] Modelo exportado en: {saved_model}")
    print(
        f"      [OK] Split cronologico: N_total={metrics['n_total']} | "
        f"N_train={metrics['n_train']} | N_test={metrics['n_test']}"
    )
    print(
        f"      [OK] Accuracy Test: {metrics['accuracy_modelo_nb']:.2%} "
        f"(vs Baseline Persistencia: {metrics['accuracy_baseline_persistencia']:.2%}, "
        f"Clase Mayoritaria: {metrics['accuracy_baseline_mayoritaria']:.2%}) | "
        f"Macro F1: {metrics['macro_f1_modelo']:.4f} | Weighted F1: {metrics['weighted_f1_modelo']:.4f} | "
        f"MAE: {metrics['mae_modelo_ugm3']} ug/m3 | RMSE: {metrics['rmse_modelo_ugm3']} ug/m3"
    )

    # Paso 4: Evaluación de las 12 preguntas del Agente (usando GROQ_API_KEY real del entorno si está disponible)
    print("\n[4/5] Evaluando banco de 12 preguntas de prueba (`src/evaluate_agent.py`)...")
    toolkit = AirQualityToolkit(
        data_csv_path=project_root / "data" / "siata_pm25_historico.csv",
        docs_rag_dir=project_root / "docs_rag",
        model_path=saved_model,
    )
    df_eval = run_agent_evaluation(toolkit=toolkit, api_key=None, t_max=5, verbose=True)
    eval_csv = project_root / "data" / "evaluacion_agente_12_preguntas.csv"
    df_eval.to_csv(eval_csv, index=False, encoding="utf-8")
    strict_hits = int((df_eval["¿Llamó Tool Correcta?"] == "Sí").sum())
    exec_mode = df_eval["Modo de Ejecución"].iloc[0] if not df_eval.empty else "N/A"
    print(f"      [OK] Modo de ejecucion: {exec_mode}")
    print(
        f"      [OK] Tool-calling estricto (igualdad de conjuntos): "
        f"{strict_hits}/{len(df_eval)} ({strict_hits / len(df_eval):.1%})"
    )
    print(f"      [OK] Evaluacion completada y guardada en: {eval_csv}")

    # Paso 4.5: Generar los 3 PDFs oficiales de entregables en `entregables_pdf/` (si matplotlib está disponible)
    print("\n[4.5/5] Verificando / generando documentos PDF oficiales en `entregables_pdf/`...")
    try:
        from src.generate_deliverables_pdf import generate_all_deliverable_pdfs

        pdfs = generate_all_deliverable_pdfs()
        for p in pdfs:
            print(f"      [OK] PDF actualizado: {p.relative_to(project_root)}")
    except ImportError:
        print("      [AVISO] `matplotlib` no esta instalado en este entorno; se conservan los PDFs existentes en `entregables_pdf/`.")
    except Exception as exc:
        print(f"      [AVISO] No se regeneraron los PDFs ({exc}); se conservan los PDFs existentes en `entregables_pdf/`.")

    # Paso 5: Lanzar Streamlit Dashboard
    print("\n[5/5] Lanzando el Dashboard interactivo en Streamlit (`src/app.py`)...")
    print("=" * 75)
    app_path = project_root / "src" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


if __name__ == "__main__":
    main()
