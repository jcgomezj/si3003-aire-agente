"""
Módulo de Análisis Exploratorio de Datos y Visualizaciones (`src/eda.py`) - SI3003 Universidad EAFIT.

Responsabilidades (Requisito Obligatorio 4 - Visualización y EDA):
1. Calcular estadísticas descriptivas por estación y municipio del Valle de Aburrá.
2. Generar visualizaciones interactivas con Plotly (para integrar en Streamlit) y exportarlas
   opcionalmente como figuras estáticas/HTML para el informe de 4 páginas:
   - Visualización 1: Serie de tiempo multivariada de PM2.5 por estaciones con bandas normativas ICA.
   - Visualización 2: Matriz de correlación de Pearson y diagrama de dispersión meteorológico
     (Efecto del lavado atmosférico por lluvia y dispersión por viento frente al PM2.5).
   - Visualización 3: Distribución comparativa (Boxplots) de PM2.5 de Norte a Sur del Valle de Aburrá.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_air_quality_data

COLOR_HEX_MAP = {
    "Buena": "#2ecc71",
    "Moderada": "#f1c40f",
    "Dañina a Grupos Sensibles": "#e67e22",
    "Dañina a la Salud": "#e74c3c",
    "Muy Dañina": "#8e44ad",
}


def compute_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla resumen de estadísticas descriptivas por estación SIATA
    (media, desviación estándar, mediana, máximo de PM2.5, lluvia media y % de días en alerta).
    """
    summary = (
        df.groupby(["station_id", "nombre", "municipio"], as_index=False)
        .agg(
            dias_monitoreados=("pm25", "count"),
            pm25_promedio=("pm25", lambda s: round(float(s.mean()), 2)),
            pm25_std=("pm25", lambda s: round(float(s.std()), 2)),
            pm25_maximo=("pm25", "max"),
            lluvia_media_mm=("precipitacion_mm", lambda s: round(float(s.mean()), 2)),
            viento_medio_ms=("velocidad_viento_ms", lambda s: round(float(s.mean()), 2)),
            pct_dias_criticos_o_alerta=(
                "pm25",
                lambda s: round(float((s > 37.0).mean() * 100.0), 1),
            ),
        )
        .sort_values("pm25_promedio", ascending=False)
    )
    return summary


def plot_pm25_timeseries_all_stations(df: pd.DataFrame) -> go.Figure:
    """
    Visualización Significativa 1: Serie de tiempo comparativa de PM2.5 en las estaciones
    del Valle de Aburrá con umbrales normativos de la Resolución 2254 de 2017.
    """
    fig = px.line(
        df.sort_values("date"),
        x="date",
        y="pm25",
        color="nombre",
        title="Evolución Temporal de PM2.5 (µg/m³) por Estación SIATA en el Valle de Aburrá",
        labels={"date": "Fecha", "pm25": "PM2.5 (µg/m³ - Promedio 24h)", "nombre": "Estación SIATA"},
        height=440,
    )
    # Bandas normativas de referencia
    fig.add_hline(
        y=12.0,
        line_dash="dot",
        line_color="#2ecc71",
        annotation_text="Umbral Buena -> Moderada (12 µg/m³)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=37.0,
        line_dash="dash",
        line_color="#e67e22",
        annotation_text="Umbral Alerta Naranja - Grupos Sensibles (37 µg/m³)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=55.4,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="Umbral Alerta Roja - Dañina a la Salud (55.4 µg/m³)",
        annotation_position="top right",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_meteorological_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Visualización Significativa 2A: Matriz de correlación entre PM2.5 actual, PM2.5 del día
    siguiente (t+1), precipitación (lluvia), velocidad del viento, temperatura y humedad relativa.
    Evidencia cuantitativamente la correlación inversa entre lluvia/viento y material particulado.
    """
    cols = [
        "pm25",
        "target_pm25_next",
        "pm25_ma3",
        "precipitacion_mm",
        "velocidad_viento_ms",
        "temperatura_c",
        "humedad_relativa_pct",
    ]
    labels = [
        "PM2.5 Hoy (t)",
        "PM2.5 Mañana (t+1)",
        "PM2.5 Media 3d",
        "Lluvia (mm)",
        "Viento (m/s)",
        "Temperatura (°C)",
        "Humedad (%)",
    ]
    corr = df[cols].dropna().corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=corr.values,
            texttemplate="%{text}",
            colorbar=dict(title="r de Pearson"),
        )
    )
    fig.update_layout(
        title="Matriz de Correlación: Contaminación (PM2.5) vs. Variables Meteorológicas (SIATA)",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_rain_wind_vs_pm25_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Visualización Significativa 2B: Análisis conjunto de Velocidad del Viento y Precipitación
    frente a la concentración de PM2.5, coloreado por categoría ICA.
    """
    fig = px.scatter(
        df,
        x="velocidad_viento_ms",
        y="pm25",
        size="precipitacion_mm",
        color="categoria_ica",
        color_discrete_map=COLOR_HEX_MAP,
        hover_name="nombre",
        hover_data=["date", "precipitacion_mm", "temperatura_c"],
        title="Efecto de Ventilación (Viento) y Lavado Atmosférico (Tamaño = Lluvia mm) sobre el PM2.5",
        labels={
            "velocidad_viento_ms": "Velocidad del Viento (m/s)",
            "pm25": "Concentración PM2.5 (µg/m³)",
            "categoria_ica": "Categoría ICA",
            "precipitacion_mm": "Lluvia (mm)",
        },
        height=430,
    )
    fig.add_hline(y=37.0, line_dash="dash", line_color="#e67e22", annotation_text="Umbral Naranja (37 µg/m³)")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def run_eda_pipeline(output_dir: Path | None = None) -> Dict[str, object]:
    """
    Ejecuta el análisis exploratorio completo y guarda tablas/gráficos en `data/eda_outputs/`
    para facilitar la redacción del informe de 4 páginas.
    """
    df = load_air_quality_data()
    summary_df = compute_eda_summary(df)
    fig_ts = plot_pm25_timeseries_all_stations(df)
    fig_corr = plot_meteorological_correlation_matrix(df)
    fig_scatter = plot_rain_wind_vs_pm25_scatter(df)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "eda_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "resumen_estadistico_estaciones.csv", index=False, encoding="utf-8")
    fig_ts.write_html(str(output_dir / "viz1_serie_tiempo_pm25.html"))
    fig_corr.write_html(str(output_dir / "viz2_matriz_correlacion_meteorologia.html"))
    fig_scatter.write_html(str(output_dir / "viz3_dispersion_viento_lluvia_pm25.html"))

    return {
        "summary_table": summary_df,
        "fig_timeseries": fig_ts,
        "fig_correlation": fig_corr,
        "fig_scatter": fig_scatter,
    }


if __name__ == "__main__":
    results = run_eda_pipeline()
    print("=== RESUMEN ESTADÍSTICO EDA POR ESTACIÓN SIATA ===")
    print(results["summary_table"].to_string(index=False))

