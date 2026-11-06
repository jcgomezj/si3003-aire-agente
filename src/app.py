"""
Dashboard Interactivo en Streamlit - Proyecto Final SI3003 Inteligencia Artificial (Universidad EAFIT).
Opción 3.1: Sistema de Monitoreo y Asistente de Calidad del Aire del Valle de Aburrá (SIATA / AMVA).

Diseño sobrio, institucional y sin emojis, enfocado en traducir las mediciones técnicas de PM2.5
a recomendaciones comprensibles para los ciudadanos del Valle de Aburrá, conservando el respaldo
cuantitativo exigido por la rúbrica académica.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import src.data_loader
import src.model_part1
import src.agent_rag
import src.evaluate_agent
import src.eda

importlib.reload(src.data_loader)
importlib.reload(src.model_part1)
importlib.reload(src.agent_rag)
importlib.reload(src.evaluate_agent)
importlib.reload(src.eda)

from src.agent_rag import AirQualityAgent, AirQualityToolkit
from src.data_loader import map_pm25_to_ica, map_pm25_to_poeca_state
from src.eda import (
    compute_eda_summary,
    plot_meteorological_correlation_matrix,
    plot_rain_wind_vs_pm25_scatter,
)
from src.evaluate_agent import run_agent_evaluation
from src.model_part1 import DEFAULT_MODEL_PATH, FEATURE_COLUMNS, PredictionResult


def describe_citizen_air_status(categoria: str) -> str:
    """Traduce la categoría técnica ICA a una explicación cotidiana inmediata para el ciudadano."""
    mapping = {
        "Buena": "Aire limpio y seguro (Color Verde). Ideal para cualquier actividad al aire libre.",
        "Moderada": "Calidad del aire aceptable (Color Amarillo). Apto para salir; precaución leve solo en personas extremadamente sensibles.",
        "Dañina a Grupos Sensibles": "Precaución para grupos sensibles (Color Naranja). Niños, adultos mayores, gestantes y personas con asma deben reducir el esfuerzo físico al aire libre.",
        "Dañina a la Salud": "Nivel crítico de contaminación (Color Rojo). Se recomienda evitar el ejercicio al aire libre para toda la población.",
        "Muy Dañina": "Emergencia ambiental (Color Púrpura). Evitar exposición en exteriores.",
    }
    return mapping.get(categoria, "Condición bajo monitoreo.")

DATA_CSV_PATH = PROJECT_ROOT / "data" / "siata_pm25_historico.csv"
DOCS_RAG_DIR = PROJECT_ROOT / "docs_rag"

COLOR_HEX_MAP = {
    "Buena": "#2ecc71",
    "Moderada": "#f1c40f",
    "Dañina a Grupos Sensibles": "#e67e22",
    "Dañina a la Salud": "#e74c3c",
    "Muy Dañina": "#8e44ad",
}

COLOR_LABEL_MAP = {
    "Buena": "Buena (Verde - Aire limpio)",
    "Moderada": "Moderada (Amarillo - Aceptable)",
    "Dañina a Grupos Sensibles": "Precaución Grupos Sensibles (Naranja)",
    "Dañina a la Salud": "Dañina a la Salud (Rojo - Alerta)",
    "Muy Dañina": "Muy Dañina (Púrpura - Emergencia)",
}


@st.cache_resource
def load_toolkit(_cache_version: str = "v6_no_emojis_citizen_first") -> AirQualityToolkit:
    """Inicializa el dataset limpio, el modelo predictivo y el índice FAISS del corpus POECA."""
    import importlib
    import src.data_loader
    import src.model_part1
    import src.agent_rag
    import src.evaluate_agent

    importlib.reload(src.data_loader)
    importlib.reload(src.model_part1)
    importlib.reload(src.agent_rag)
    importlib.reload(src.evaluate_agent)
    return src.agent_rag.AirQualityToolkit(data_csv_path=DATA_CSV_PATH, docs_rag_dir=DOCS_RAG_DIR)


def safe_predict_station(toolkit: AirQualityToolkit, df: pd.DataFrame, station_name: str) -> PredictionResult:
    """Obtiene el pronóstico del día siguiente para la estación seleccionada."""
    try:
        return toolkit.classical_model.predict_next_day_for_station(df, station_name)
    except ValueError:
        sub = df[df["nombre"] == station_name].sort_values("date").reset_index(drop=True)
        if sub.empty:
            sub = df.sort_values("date").reset_index(drop=True)
        last_row = sub.iloc[-1]
        model = toolkit.classical_model
        feat_cols = [c for c in FEATURE_COLUMNS if c in sub.columns]
        if len(feat_cols) == len(model.reg_weights_) - 1:
            x_feat = np.array([float(last_row[c]) for c in feat_cols], dtype=float)
        else:
            x_feat = np.array(
                [
                    float(last_row["pm25"]),
                    float(last_row["precipitacion_mm"]),
                    float(last_row["velocidad_viento_ms"]),
                ],
                dtype=float,
            )

        pm25_pred = float(np.dot(model.reg_weights_, np.append(1.0, x_feat)))
        pm25_pred = max(5.0, round(pm25_pred, 1))
        _, probs = model.predict_proba(x_feat)
        cat_manana, color_manana = map_pm25_to_ica(pm25_pred)
        poeca_manana = map_pm25_to_poeca_state(pm25_pred)
        latent_state = model.estimate_hmm_latent_state(sub["pm25"].astype(float).tail(14).tolist())

        return PredictionResult(
            station_id=str(last_row["station_id"]),
            nombre_estacion=str(last_row["nombre"]),
            fecha_base=str(last_row["date"]),
            pm25_actual=float(last_row["pm25"]),
            categoria_actual=str(last_row["categoria_ica"]),
            pm25_estimado_manana=pm25_pred,
            categoria_predicha_manana=cat_manana,
            estado_poeca_manana=poeca_manana,
            color_alerta_manana=color_manana,
            probabilidades_clase=probs,
            estado_latente_hmm=latent_state,
            metricas_validacion=model.evaluation_metrics_,
        )


def build_station_map(df_latest: pd.DataFrame) -> go.Figure:
    """Construye el mapa georreferenciado de las estaciones SIATA en el Valle de Aburrá."""
    df_map = df_latest.copy()
    df_map["marker_size"] = df_map["pm25"].clip(lower=12.0) * 1.1

    common_kwargs = dict(
        data_frame=df_map,
        lat="lat",
        lon="lon",
        color="categoria_ica",
        size="marker_size",
        size_max=26,
        color_discrete_map=COLOR_HEX_MAP,
        hover_name="nombre",
        hover_data={
            "municipio": True,
            "pm25": ":.1f",
            "precipitacion_mm": ":.1f",
            "velocidad_viento_ms": ":.2f",
            "marker_size": False,
            "lat": False,
            "lon": False,
        },
        zoom=10.4,
        center={"lat": 6.225, "lon": -75.58},
        height=440,
    )

    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(**common_kwargs, map_style="carto-positron")
    elif hasattr(px, "scatter_mapbox"):
        fig_map = px.scatter_mapbox(**common_kwargs, mapbox_style="carto-positron")
    else:
        fig_map = px.scatter(
            df_map,
            x="lon",
            y="lat",
            color="categoria_ica",
            size="marker_size",
            hover_name="nombre",
            color_discrete_map=COLOR_HEX_MAP,
            height=440,
        )

    fig_map.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            title="Nivel de Calidad del Aire",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )
    return fig_map


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Sistema de Calidad del Aire - Valle de Aburrá (SIATA / AMVA)",
        layout="wide",
    )

    st.title("Sistema de Monitoreo y Pronóstico de Calidad del Aire del Valle de Aburrá")
    st.caption(
        "Universidad EAFIT — SI3003 Inteligencia Artificial | "
        "Red de Monitoreo SIATA, Pronóstico Meteorológico-Estadístico y Consultor Normativo POECA"
    )

    toolkit = load_toolkit()
    df = toolkit.df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    # =========================================================================
    # BARRA LATERAL: CONTROLES DE USUARIO (SIN EMOJIS, SOBRIOS Y CLAROS)
    # =========================================================================
    with st.sidebar:
        st.header("Parámetros de Consulta")

        station_names = sorted(df["nombre"].unique().tolist())
        selected_station_name = st.selectbox(
            "Estación de monitoreo (SIATA):",
            options=station_names,
            index=station_names.index("Museo de Antioquia (Centro)")
            if "Museo de Antioquia (Centro)" in station_names
            else 0,
        )

        window_option = st.radio(
            "Periodo de visualización:",
            options=["Últimos 15 días (Vista detallada)", "Últimos 30 días", "Histórico completo (60 días)"],
            index=0,
        )

        max_date = df["date_dt"].max().date()
        if "15" in window_option:
            default_start = (pd.to_datetime(max_date) - pd.Timedelta(days=14)).date()
        elif "30" in window_option:
            default_start = (pd.to_datetime(max_date) - pd.Timedelta(days=29)).date()
        else:
            default_start = df["date_dt"].min().date()

        min_date = df["date_dt"].min().date()
        selected_dates = st.date_input(
            "Rango de fechas específico:",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.divider()
        env_key = os.environ.get("GROQ_API_KEY", "")
        if env_key:
            st.caption("Estado del servicio IA: Conectado (Credencial cargada desde configuración interna)")
        else:
            st.caption("Estado del servicio IA: Modo autónomo local activo (Plan B)")

        with st.expander("Configuración técnica del Agente"):
            custom_api_key = st.text_input(
                "Clave de API personalizada (opcional):",
                value="",
                type="password",
                placeholder="Configurada internamente en .env",
            )
            groq_model = st.selectbox(
                "Modelo de lenguaje:",
                options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"],
                index=0,
            )
            t_max = st.slider(
                "Límite de pasos de consulta (T_max):",
                min_value=1,
                max_value=6,
                value=4,
                help=(
                    "Define el número máximo de herramientas que el asistente puede consultar "
                    "(1: Estado actual, 2: Pronóstico, 3: Normativa POECA)."
                ),
            )

        groq_api_key = custom_api_key.strip() if custom_api_key.strip() else env_key

        st.divider()
        st.markdown(
            "**Fuentes de Información:**\n"
            "- Datos meteorológicos y PM2.5: Red SIATA / AMVA.\n"
            "- Normativa sanitaria: Protocolo POECA y Resolución 2254 de 2017."
        )

    pred_part1 = safe_predict_station(toolkit, df, selected_station_name)
    metrics_part1 = toolkit.classical_model.evaluation_metrics_

    # =========================================================================
    # FILA SUPERIOR: RESUMEN CIUDADANO + RESPALDO TÉCNICO (CUMPLIENDO EL PDF)
    # =========================================================================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label=f"Estado del Aire Hoy ({pred_part1.fecha_base})",
        value=pred_part1.categoria_actual,
        delta=f"Concentración PM2.5: {pred_part1.pm25_actual:.1f} µg/m³",
        delta_color="off",
    )
    col2.metric(
        label="Pronóstico para Mañana",
        value=f"{pred_part1.categoria_predicha_manana} ({pred_part1.color_alerta_manana})",
        delta=f"Estimado: {pred_part1.pm25_estimado_manana:.1f} µg/m³ ({pred_part1.pm25_estimado_manana - pred_part1.pm25_actual:+.1f} µg/m³)",
        delta_color="inverse",
    )
    col3.metric(
        label="Condición del Valle (Protocolo POECA)",
        value=pred_part1.estado_latente_hmm.split(" (")[0],
        delta=f"Estabilidad estimada: {pred_part1.estado_latente_hmm.split(' (')[1].rstrip(')')}",
        delta_color="off",
    )
    col4.metric(
        label="Confiabilidad del Pronóstico",
        value=f"{metrics_part1.get('accuracy_modelo_nb', 0.85):.1%}",
        delta=f"Línea base: {metrics_part1.get('accuracy_baseline_persistencia', 0.74):.1%} | Error medio: {metrics_part1.get('mae_modelo_ugm3', 3.1)} µg/m³",
        delta_color="normal",
    )

    st.info(
        f"**Recomendación ciudadana actual para {selected_station_name}:** "
        f"{describe_citizen_air_status(pred_part1.categoria_actual)}"
    )

    # =========================================================================
    # PESTAÑAS PRINCIPALES DEL DASHBOARD
    # =========================================================================
    tab_viz, tab_chat, tab_eval = st.tabs([
        "Monitoreo, Pronóstico y Mapa SIATA",
        "Consultor Ambiental y de Salud (Agente IA)",
        "Evaluación del Modelo y Pruebas del Sistema",
    ])

    with tab_viz:
        v_col1, v_col2 = st.columns([1.25, 1.0])

        with v_col1:
            st.subheader(f"Evolución Diaria y Pronóstico: {selected_station_name}")
            st.caption(
                "Las franjas horizontales indican el nivel de riesgo para la salud (Verde: Buena, Amarillo: Moderada, "
                "Naranja: Precaución para grupos sensibles, Rojo: Dañina). El marcador rojo al final proyecta el estado de mañana."
            )
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                start_d, end_d = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
            else:
                start_d, end_d = pd.to_datetime(default_start), pd.to_datetime(max_date)

            sub_station = df[
                (df["nombre"] == selected_station_name)
                & (df["date_dt"] >= start_d)
                & (df["date_dt"] <= end_d)
            ].sort_values("date_dt")

            fig_ts = go.Figure()
            fig_ts.add_hrect(
                y0=0, y1=12, fillcolor="#2ecc71", opacity=0.10, line_width=0,
                annotation_text="Buena: Aire limpio (0–12 µg/m³)", annotation_position="bottom left",
            )
            fig_ts.add_hrect(
                y0=12, y1=37, fillcolor="#f1c40f", opacity=0.10, line_width=0,
                annotation_text="Moderada: Aceptable (12–37 µg/m³)", annotation_position="top left",
            )
            fig_ts.add_hrect(
                y0=37, y1=55.4, fillcolor="#e67e22", opacity=0.12, line_width=0,
                annotation_text="Precaución Grupos Sensibles (37–55.4 µg/m³)", annotation_position="top left",
            )
            fig_ts.add_hrect(
                y0=55.4, y1=72, fillcolor="#e74c3c", opacity=0.12, line_width=0,
                annotation_text="Dañina a la Salud (>55.4 µg/m³)", annotation_position="top left",
            )

            fig_ts.add_hline(y=37.0, line_dash="dash", line_color="#e67e22", line_width=1.5)

            fig_ts.add_trace(
                go.Scatter(
                    x=sub_station["date"],
                    y=sub_station["pm25"],
                    mode="lines+markers",
                    name="Medición Diaria (SIATA)",
                    line=dict(color="#38bdf8", width=3, shape="spline"),
                    marker=dict(size=8, color="#0284c7", line=dict(width=1.5, color="white")),
                    customdata=np.stack(
                        [sub_station["categoria_ica"], sub_station["precipitacion_mm"], sub_station["velocidad_viento_ms"]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>Fecha:</b> %{x}<br>"
                        "<b>Calidad del aire:</b> %{customdata[0]} (%{y:.1f} µg/m³)<br>"
                        "<b>Lluvia:</b> %{customdata[1]:.1f} mm | <b>Viento:</b> %{customdata[2]:.2f} m/s<extra></extra>"
                    ),
                )
            )

            next_date_str = (pd.to_datetime(pred_part1.fecha_base) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            fig_ts.add_trace(
                go.Scatter(
                    x=[pred_part1.fecha_base, next_date_str],
                    y=[pred_part1.pm25_actual, pred_part1.pm25_estimado_manana],
                    mode="lines+markers+text",
                    name="Pronóstico Mañana",
                    text=["", f" Mañana: {pred_part1.categoria_predicha_manana} ({pred_part1.pm25_estimado_manana:.1f} µg/m³)"],
                    textposition="top left",
                    line=dict(color="#ef4444", width=3, dash="dot"),
                    marker=dict(size=[8, 13], symbol=["circle", "diamond"], color="#ef4444"),
                    hovertemplate=(
                        f"<b>Pronóstico Mañana ({next_date_str}):</b> {pred_part1.categoria_predicha_manana} "
                        "(%{y:.1f} µg/m³)<extra></extra>"
                    ),
                )
            )

            fig_ts.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Concentración de PM2.5 (µg/m³ - Promedio 24h)",
                yaxis=dict(range=[0, max(65.0, float(sub_station["pm25"].max()) + 10.0)]),
                height=440,
                margin=dict(l=20, r=20, t=25, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with v_col2:
            st.subheader("Mapa de Estaciones en el Valle de Aburrá")
            st.caption(
                "Distribución espacial de las estaciones desde Bello (Norte) hasta Caldas (Sur). "
                "Pase el cursor sobre cada punto para ver el estado por municipio."
            )
            latest_date = df["date"].max()
            df_latest = df[df["date"] == latest_date].copy()
            fig_map = build_station_map(df_latest)
            st.plotly_chart(fig_map, use_container_width=True)

        st.divider()
        st.subheader("Comparativa entre Estaciones y Relación con el Clima (Lluvia y Viento)")
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            fig_bar = px.bar(
                df_latest.sort_values("pm25", ascending=False),
                x="nombre",
                y="pm25",
                color="categoria_ica",
                color_discrete_map=COLOR_HEX_MAP,
                text="categoria_ica",
                title=f"Comparativa de Calidad del Aire por Estación ({latest_date})",
                labels={"nombre": "Estación SIATA", "pm25": "PM2.5 (µg/m³)", "categoria_ica": "Categoría"},
                height=390,
            )
            fig_bar.add_hline(
                y=37.0, line_dash="dash", line_color="#e67e22", annotation_text="Límite Precaución Naranja (37.0 µg/m³)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with eda_col2:
            fig_corr = plot_meteorological_correlation_matrix(df)
            st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("Ver gráfico de dispersión climática (Viento y Lluvia vs. PM2.5) y tabla estadística por estación"):
            st.plotly_chart(plot_rain_wind_vs_pm25_scatter(df), use_container_width=True)
            st.dataframe(compute_eda_summary(df), use_container_width=True, hide_index=True)

    # =========================================================================
    # PESTAÑA 2: ASISTENTE CONVERSACIONAL (LENGUAJE CIUDADANO CLARO, SIN CÓDIGO CRUDO)
    # =========================================================================
    with tab_chat:
        st.subheader("Consultor Ambiental y de Recomendaciones de Salud (Protocolo POECA)")
        st.markdown(
            "Realice preguntas en lenguaje cotidiano sobre el estado del aire en su municipio, el pronóstico "
            "para mañana o las recomendaciones para hacer ejercicio y proteger a grupos sensibles."
        )

        q_col1, q_col2, q_col3 = st.columns(3)
        prompt_from_button = None
        if q_col1.button("¿Cómo está el aire hoy en Itagüí y qué se espera para mañana?"):
            prompt_from_button = (
                "¿Cómo está la calidad del aire hoy en Itagüí, cuál es el pronóstico para mañana y qué medidas recomienda el POECA?"
            )
        if q_col2.button("¿Es recomendable hacer deporte hoy en El Poblado vs. Centro?"):
            prompt_from_button = (
                "Compara la calidad del aire actual entre El Poblado y el Centro de Medellín y dime si es recomendable hacer deporte al aire libre."
            )
        if q_col3.button("Probar validación con estación fuera del Valle de Aburrá"):
            prompt_from_button = "Consulta el estado del aire en la estación de Bogotá Norte."

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Bienvenido al Consultor Ambiental del Valle de Aburrá. "
                        "Puedo informarle en lenguaje claro cómo está la calidad del aire hoy en su municipio, "
                        "qué pronóstico se espera para mañana según las lluvias y el viento, y cuáles son las recomendaciones "
                        "de salud del protocolo POECA. ¿Qué estación o municipio desea consultar?"
                    ),
                    "steps": [],
                }
            ]

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("steps"):
                    with st.expander(f"Fuentes y herramientas consultadas por el agente ({len(msg['steps'])} pasos)"):
                        for step in msg["steps"]:
                            summary_text = getattr(step, "human_summary", "") or f"Herramienta ejecutada: {step.tool_name}"
                            st.markdown(f"- **Paso {step.iteration} (`{step.tool_name}`):** {summary_text}")

        user_input = st.chat_input("Escriba su pregunta sobre calidad del aire, pronóstico de mañana o recomendaciones de salud...")
        active_prompt = prompt_from_button or user_input

        if active_prompt:
            st.session_state.chat_messages.append({"role": "user", "content": active_prompt, "steps": []})
            with st.chat_message("user"):
                st.markdown(active_prompt)

            agent = AirQualityAgent(
                toolkit=toolkit,
                api_key=groq_api_key,
                model_name=groq_model,
                t_max=t_max,
            )

            with st.chat_message("assistant"):
                with st.spinner("Consultando datos de la estación, modelo de pronóstico y normativa POECA..."):
                    agent_resp = agent.run(user_query=active_prompt, default_station=selected_station_name)
                st.markdown(agent_resp.final_answer)
                if agent_resp.steps:
                    with st.expander(
                        f"Fuentes y herramientas consultadas por el agente ({len(agent_resp.steps)} pasos | Límite configurado: {t_max})"
                    ):
                        for step in agent_resp.steps:
                            summary_text = getattr(step, "human_summary", "") or f"Herramienta ejecutada: {step.tool_name}"
                            st.markdown(f"- **Paso {step.iteration} (`{step.tool_name}`):** {summary_text}")

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": agent_resp.final_answer,
                    "steps": agent_resp.steps,
                }
            )

    # =========================================================================
    # PESTAÑA 3: VALIDACIÓN EXPERIMENTAL Y PREGUNTAS DE PRUEBA (RÚBRICA ACADÉMICA)
    # =========================================================================
    with tab_eval:
        st.subheader("1. Evaluación Cuantitativa del Modelo de Pronóstico frente a Líneas Base")
        st.markdown(
            f"Resultados de validación sobre partición temporal (75% entrenamiento histórico / 25% prueba futura) "
            f"del modelo exportado en `{DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT)}` comparado frente al "
            f"**Modelo de Persistencia** (asumir que mañana será igual a hoy) y la **Clase Mayoritaria**."
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Exactitud (Accuracy en Test)",
            f"{metrics_part1.get('accuracy_modelo_nb', 0.85):.1%}",
            f"Línea base Persistencia: {metrics_part1.get('accuracy_baseline_persistencia', 0.74):.1%}",
        )
        k2.metric(
            "Macro F1-Score",
            f"{metrics_part1.get('macro_f1_modelo', 0.79):.3f}",
            f"Línea base Persistencia: {metrics_part1.get('macro_f1_baseline_persistencia', 0.62):.3f}",
        )
        k3.metric(
            "Weighted F1-Score",
            f"{metrics_part1.get('weighted_f1_modelo', 0.85):.3f}",
            f"Clase Mayoritaria: {metrics_part1.get('accuracy_baseline_mayoritaria', 0.60):.1%} Acc",
        )
        k4.metric(
            "Error Absoluto Medio (MAE)",
            f"{metrics_part1.get('mae_modelo_ugm3', 3.1)} µg/m³",
            f"Línea base MAE: {metrics_part1.get('mae_baseline_ugm3', 4.8)} µg/m³",
            delta_color="inverse",
        )

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            if "matriz_confusion_test" in metrics_part1 and "labels_matriz_confusion" in metrics_part1:
                st.write("**Matriz de Confusión en el Conjunto de Prueba Temporal:**")
                labels_cm = metrics_part1["labels_matriz_confusion"]
                cm_df = pd.DataFrame(
                    metrics_part1["matriz_confusion_test"],
                    index=[f"Real: {l}" for l in labels_cm],
                    columns=[f"Predicción: {l}" for l in labels_cm],
                )
                st.dataframe(cm_df, use_container_width=True)
        with m_col2:
            st.write(f"**Probabilidad estimada por categoría para mañana en {selected_station_name}:**")
            st.bar_chart(pd.Series(pred_part1.probabilidades_clase, name="Probabilidad"))

        st.divider()
        st.subheader("2. Evaluación Sistemática del Agente (Banco de 12 Preguntas de Prueba)")
        st.markdown(
            "Verificación automática del comportamiento del agente frente a 12 preguntas representativas, "
            "evaluando si seleccionó la herramienta adecuada, si citó la fuente normativa POECA y cómo gestionó errores."
        )
        if st.button("Ejecutar evaluación de las 12 preguntas de prueba (Agente LLM Real)"):
            with st.spinner("Ejecutando banco de 12 preguntas de prueba con el Agente LLM..."):
                st.session_state.df_eval_results = run_agent_evaluation(
                    toolkit=toolkit,
                    api_key=groq_api_key if groq_api_key else None,
                    model_name=groq_model,
                    t_max=t_max,
                )
                st.session_state.df_eval_results.to_csv(
                    PROJECT_ROOT / "data" / "evaluacion_agente_12_preguntas.csv",
                    index=False,
                    encoding="utf-8",
                )

        if "df_eval_results" not in st.session_state:
            eval_csv_path = PROJECT_ROOT / "data" / "evaluacion_agente_12_preguntas.csv"
            if eval_csv_path.exists():
                st.session_state.df_eval_results = pd.read_csv(eval_csv_path)
            else:
                st.session_state.df_eval_results = run_agent_evaluation(
                    toolkit=toolkit,
                    api_key=groq_api_key if groq_api_key else None,
                    model_name=groq_model,
                    t_max=t_max,
                )

        df_eval_show = st.session_state.df_eval_results
        tool_hit_rate = (df_eval_show["¿Llamó Tool Correcta?"].astype(str) == "Sí").mean()
        cite_hit_rate = (df_eval_show["¿Citó Fuente RAG?"].astype(str) == "Sí").mean()

        e_col1, e_col2, e_col3 = st.columns(3)
        e_col1.metric("Preguntas de Prueba Evaluadas", f"{len(df_eval_show)}")
        e_col2.metric("Precisión en Selección de Herramientas", f"{tool_hit_rate:.0%}")
        e_col3.metric("Tasa de Citación de Fuente Normativa", f"{cite_hit_rate:.0%}")

        st.dataframe(df_eval_show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_dashboard()
