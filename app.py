"""Dashboard interactivo — Agente de calidad del aire del Valle de Aburrá.

Ejecutar con:
    streamlit run app.py

Cumple los requisitos del enunciado:
  - Selección de parámetros por el usuario (estación, rango de fechas).
  - >= 2 visualizaciones significativas (mapa de estaciones + serie de tiempo).
  - Chat conversacional con el agente LLM (loop + RAG + herramientas).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src import config
from src.data.loader import list_estaciones, load_air_quality

st.set_page_config(page_title="Aire Valle de Aburrá", page_icon="🌫️", layout="wide")


@st.cache_data
def _load() -> pd.DataFrame:
    return load_air_quality()


df = _load()

# ---------------------------------------------------------------------------
# Barra lateral: selección de parámetros
# ---------------------------------------------------------------------------
st.sidebar.title("🌫️ Parámetros")
estacion = st.sidebar.selectbox("Estación", list_estaciones())
fecha_min, fecha_max = df["fecha"].min(), df["fecha"].max()
rango = st.sidebar.slider(
    "Rango de fechas",
    min_value=fecha_min.to_pydatetime(),
    max_value=fecha_max.to_pydatetime(),
    value=(fecha_min.to_pydatetime(), fecha_max.to_pydatetime()),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Modelo LLM: `%s`\n\nProveedor: `%s`" % (config.LLM_MODEL, config.LLM_BASE_URL)
)

st.title("Agente de calidad del aire del Valle de Aburrá")
st.caption(
    "Proyecto final SI3003 — Parte 1 (modelo clásico) + Parte 2 (agente LLM con RAG)."
)

# ---------------------------------------------------------------------------
# Visualización 1: mapa de estaciones (último valor de PM2.5)
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mapa de estaciones")
    ult = df.sort_values("fecha").groupby("estacion").tail(1).copy()
    ult["radio"] = ult["pm25"] * 30
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=ult,
        get_position=["lon", "lat"],
        get_radius="radio",
        get_fill_color=[200, 30, 0, 140],
        pickable=True,
    )
    view = pdk.ViewState(latitude=6.22, longitude=-75.58, zoom=10)
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{estacion}\nPM2.5: {pm25}\n{categoria}"},
        )
    )

# ---------------------------------------------------------------------------
# Visualización 2: serie de tiempo de PM2.5 para la estación seleccionada
# ---------------------------------------------------------------------------
with col2:
    st.subheader(f"Serie de tiempo — {estacion}")
    mask = (
        (df["estacion"] == estacion)
        & (df["fecha"] >= pd.Timestamp(rango[0]))
        & (df["fecha"] <= pd.Timestamp(rango[1]))
    )
    serie = df[mask]
    fig = px.line(serie, x="fecha", y="pm25", markers=True, labels={"pm25": "PM2.5 (µg/m³)"})
    fig.add_hline(y=37, line_dash="dash", annotation_text="Umbral grupos sensibles")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chat con el agente
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Chat con el agente")

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Pregúntale al agente sobre la calidad del aire...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            from src.agent.agent import run_agent

            with st.spinner("Pensando y consultando herramientas..."):
                result = run_agent(prompt)
            st.markdown(result.answer)
            if result.tools_used:
                with st.expander("🔧 Herramientas usadas / traza"):
                    st.write("Herramientas:", result.tools_used)
                    st.json(result.steps)
            st.session_state.chat.append(
                {"role": "assistant", "content": result.answer}
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"⚠️ No pude ejecutar el agente: {exc}\n\n"
                "Verifica que configuraste tu `LLM_API_KEY` en `.env` "
                "(ver `.env.example`)."
            )
            st.error(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})
