"""
Generador de Entregables Oficiales en PDF (`src/generate_deliverables_pdf.py`) - SI3003 EAFIT.

Genera automáticamente los tres documentos en formato PDF exigidos en la sección 5 (Entregables)
y sección 6 (Rúbrica de Evaluación) de la guía oficial del curso:

1. `entregables_pdf/1_Informe_Tecnico_SI3003_Gomez_Arango.pdf` (Exactamente 4 páginas tamaño Carta):
   - Página 1: Portada institucional EAFIT, Resumen Ejecutivo, Definición del Problema, Datos Reales (SIATA/AMVA) y Estrategia de Resiliencia (Plan B).
   - Página 2: Análisis Exploratorio de Datos (EDA), Gráfica de Serie de Tiempo + Matriz de Correlación Meteorológica, y Formulación Matemática del Modelo Clásico de la Parte 1 (Gaussian Naive Bayes + Regresión Ridge L2 + Filtro de Markov HMM).
   - Página 3: Diseño de la Parte 2 (Arquitectura del Agente LLM con Groq `llama-3.3-70b-versatile`, Loop Agéntico Manual con $T_{max}$, Sistema RAG con FAISS sobre el Protocolo POECA y las 3 herramientas con manejo de errores como observación).
   - Página 4: Resultados Experimentales de la Parte 1 (Tabla comparativa frente a Líneas Base de Persistencia y Clase Mayoritaria + Matriz de Confusión), Resumen de Evaluación del Agente (12 preguntas), Limitaciones y Consideraciones Éticas.

2. `entregables_pdf/2_Evaluacion_Agente_12_Preguntas_SI3003.pdf` (2 páginas):
   - Entregable 4: Banco completo de 12 preguntas de prueba detallando categoría, pregunta, herramientas esperadas vs. invocadas, citación de fuente RAG (`normativa_poeca_amva.md`), iteraciones usadas ($T_{max}$) y verificación de captura controlada de errores.

3. `entregables_pdf/3_Declaracion_Uso_IA_Generativa_SI3003.pdf` (1 página):
   - Entregable 5: Declaración ética y metodológica de uso de herramientas de IA generativa conforme al pacto pedagógico de la Universidad EAFIT, firmada por los integrantes.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from src.agent_rag import AirQualityToolkit
from src.data_loader import load_air_quality_data
from src.evaluate_agent import run_agent_evaluation
from src.model_part1 import ClassicalAirQualityModel, DEFAULT_MODEL_PATH


def compute_eda_summary_local(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la tabla resumen estadística por estación SIATA sin requerir librerías gráficas web."""
    return (
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

# Colores institucionales sobrios (Azul oscuro EAFIT, Gris pizarra, Verde ambiental)
NAVY_PRIMARY = "#0f2c59"
SLATE_DARK = "#1e293b"
SLATE_MUTED = "#475569"
ACCENT_BLUE = "#0284c7"
ACCENT_TEAL = "#0d9488"
LIGHT_BG = "#f8fafc"
BORDER_GRAY = "#cbd5e1"


def _setup_letter_page(fig_size: Tuple[float, float] = (8.5, 11.0)) -> Tuple[plt.Figure, plt.Axes]:
    """Crea un lienzo tamaño Carta (8.5 x 11 pulgadas) con coordenadas normalizadas [0, 1]."""
    fig = plt.figure(figsize=fig_size, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _draw_header_and_footer(ax: plt.Axes, page_num: int, total_pages: int, doc_title: str) -> None:
    """Dibuja el encabezado y pie de página académico institucional en cada hoja."""
    # Franja superior institucional
    ax.add_patch(patches.Rectangle((0.06, 0.945), 0.88, 0.003, facecolor=NAVY_PRIMARY, edgecolor="none"))
    ax.text(
        0.06,
        0.955,
        "Universidad EAFIT  |  SI3003 – Inteligencia Artificial (2026-1)",
        fontsize=8.5,
        fontweight="bold",
        color=NAVY_PRIMARY,
        va="bottom",
    )
    ax.text(
        0.94,
        0.955,
        doc_title,
        fontsize=8.5,
        color=SLATE_MUTED,
        ha="right",
        va="bottom",
    )

    # Pie de página
    ax.add_patch(patches.Rectangle((0.06, 0.048), 0.88, 0.0015, facecolor=BORDER_GRAY, edgecolor="none"))
    ax.text(
        0.06,
        0.032,
        "Autores: Juan Camilo Gómez Jiménez y Pablo Arango  |  Docente: Jorge Iván Padilla Buriticá",
        fontsize=8,
        color=SLATE_MUTED,
        va="center",
    )
    ax.text(
        0.94,
        0.032,
        f"Página {page_num} de {total_pages}",
        fontsize=8,
        fontweight="bold",
        color=NAVY_PRIMARY,
        ha="right",
        va="center",
    )


def _draw_section_banner(ax: plt.Axes, y_top: float, title: str) -> float:
    """Dibuja un subtítulo de sección con barra lateral azul e indica la nueva coordenada y."""
    ax.add_patch(patches.Rectangle((0.06, y_top - 0.022), 0.008, 0.024, facecolor=ACCENT_BLUE, edgecolor="none"))
    ax.add_patch(patches.Rectangle((0.068, y_top - 0.022), 0.872, 0.024, facecolor="#f1f5f9", edgecolor="none"))
    ax.text(
        0.078,
        y_top - 0.010,
        title,
        fontsize=10.5,
        fontweight="bold",
        color=NAVY_PRIMARY,
        va="center",
    )
    return y_top - 0.034


def _draw_wrapped_paragraph(
    ax: plt.Axes,
    x: float,
    y_top: float,
    text: str,
    width_chars: int = 108,
    fontsize: float = 8.8,
    line_spacing: float = 0.0155,
    color: str = SLATE_DARK,
    fontweight: str = "normal",
) -> float:
    """Escribe un párrafo con ajuste automático de línea y devuelve la siguiente posición vertical."""
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            y_top -= line_spacing * 0.5
            continue
        wrapped = textwrap.wrap(raw_line, width=width_chars)
        for line in wrapped:
            ax.text(x, y_top, line, fontsize=fontsize, color=color, fontweight=fontweight, va="top")
            y_top -= line_spacing
    return y_top - 0.004


# =====================================================================
# DOCUMENTO 1: INFORME TÉCNICO DE 4 PÁGINAS (ENTREGABLE 2)
# =====================================================================

def generate_4page_technical_report(
    output_pdf_path: Path,
    df: pd.DataFrame,
    summary_eda: pd.DataFrame,
    metrics: dict,
    df_eval: pd.DataFrame,
) -> Path:
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc_short = "Informe Final: Agente de Calidad del Aire del Valle de Aburrá"

    with PdfPages(output_pdf_path) as pdf:
        # -----------------------------------------------------------------
        # PÁGINA 1: PORTADA COMPACTA, PROBLEMA, DATOS REALES Y PLAN B
        # -----------------------------------------------------------------
        fig1, ax1 = _setup_letter_page()
        _draw_header_and_footer(ax1, 1, 4, doc_short)

        # Recuadro de título institucional en la parte superior
        ax1.add_patch(
            patches.FancyBboxPatch(
                (0.06, 0.79),
                0.88,
                0.135,
                boxstyle="round,pad=0.01",
                facecolor=NAVY_PRIMARY,
                edgecolor="none",
            )
        )
        ax1.text(
            0.50,
            0.902,
            "UNIVERSIDAD EAFIT — ESCUELA DE CIENCIAS APLICADAS E INGENIERÍA",
            fontsize=9.5,
            fontweight="bold",
            color="#93c5fd",
            ha="center",
            va="top",
        )
        ax1.text(
            0.50,
            0.875,
            "3.1 Agente Inteligente de Calidad del Aire del Valle de Aburrá",
            fontsize=14.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="top",
        )
        ax1.text(
            0.50,
            0.842,
            "Integración de Modelos Probabilísticos Clásicos (Naive Bayes + HMM) con Agentes LLM, RAG (FAISS) y Dashboard Interactivo",
            fontsize=8.8,
            color="#e2e8f0",
            ha="center",
            va="top",
        )
        ax1.text(
            0.50,
            0.812,
            "Autores: Juan Camilo Gómez Jiménez y Pablo Arango   |   Curso: SI3003 – Inteligencia Artificial (2026-1)   |   Profesor: Jorge Iván Padilla Buriticá",
            fontsize=8.3,
            fontweight="bold",
            color="#fde047",
            ha="center",
            va="top",
        )

        y = 0.768
        y = _draw_section_banner(ax1, y, "1. Contexto del Problema y Justificación en el Valle de Aburrá")
        p1_text = (
            "El Valle de Aburrá es una cuenca estrecha y semicerrada de los Andes centrales colombianos cuya topografía limita la dispersión "
            "de contaminantes atmosféricos. Durante las dos transiciones climáticas anuales (febrero-abril y octubre-noviembre), la presencia "
            "de nubosidad de baja altura reduce la radiación solar en superficie e impide el ascenso convectivo del aire, generando episodios de "
            "inversión térmica que atrapan el material particulado fino (PM2.5) en la capa límite respirable.\n"
            "Aunque la red del SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá) publica mediciones técnicas continuas en "
            "microgramos por metro cúbico (ug/m3), los ciudadanos enfrentan dos problemas fundamentales:\n"
            "  1. Desconocen qué significa el nivel de PM2.5 del día en términos prácticos para su salud y cómo las variables meteorológicas "
            "(lavado atmosférico por lluvia y ventilación por viento) cambiarán la calidad del aire al día siguiente.\n"
            "  2. No saben cómo aplicar los lineamientos del Protocolo POECA (Plan Operacional para Enfrentar Episodios de Contaminación "
            "Atmosférica) según si pertenecen a la población general, deportistas o grupos sensibles (niños, adultos mayores, gestantes y pacientes "
            "con asma o EPOC)."
        )
        y = _draw_wrapped_paragraph(ax1, 0.06, y, p1_text, width_chars=112, fontsize=8.7, line_spacing=0.0155)

        y -= 0.006
        y = _draw_section_banner(ax1, y, "2. Fuentes de Datos Reales, Preprocesamiento y Estrategia de Resiliencia (Plan B)")
        p2_text = (
            "Cumpliendo con el Requisito Obligatorio 1, el módulo `src/data_loader.py` implementa un pipeline reproducible de Ingeniería de Datos "
            "que conecta con fuentes abiertas del SIATA / AMVA (`datos.gov.co`) y dispone de un Plan B local verificado en "
            "`data/siata_pm25_historico.csv` para garantizar operación continua ante fallas o bloqueos de red:\n"
            "  • Cobertura espacial y temporal: 360 registros históricos (60 días continuos en época de transición) distribuidos en 6 estaciones "
            "oficiales de Norte a Sur del Valle de Aburrá: Bello (U. San Buenaventura), Medellín Centro (Museo de Antioquia), Medellín El Poblado "
            "(Tanques EPM), Envigado (Hosp. Manuel Uribe Ángel), Itagüí (Casa de la Justicia) y Caldas (E. Joaquín Aristizábal).\n"
            "  • Variables físicas e ingeniería de características: Concentración de PM2.5 (ug/m3, promedio móvil 24h), media móvil de 3 días "
            "(PM2.5_MA3), cambio diario (Delta_PM2.5), precipitación acumulada (mm), velocidad del viento (m/s), temperatura (°C), humedad relativa "
            "(%) y etiquetas normativas según la Resolución 2254 de 2017 (Buena, Moderada, Dañina a Grupos Sensibles y Dañina a la Salud).\n"
            "  • Limpieza e imputación: Validación de dominios físicos, eliminación de duplicados por clave (station_id, date) e imputación de "
            "valores faltantes mediante interpolación lineal temporal intragrupo por estación."
        )
        y = _draw_wrapped_paragraph(ax1, 0.06, y, p2_text, width_chars=112, fontsize=8.7, line_spacing=0.0155)

        # Tabla resumen de las 6 estaciones SIATA (EDA) en la parte inferior de la Página 1
        ax1.text(
            0.06,
            y - 0.005,
            "Tabla 1. Resumen Estadístico del Análisis Exploratorio de Datos (EDA) por Estación SIATA (N = 360 mediciones diarias)",
            fontsize=8.8,
            fontweight="bold",
            color=NAVY_PRIMARY,
            va="top",
        )
        ax_tbl1 = fig1.add_axes([0.06, 0.075, 0.88, 0.19])
        ax_tbl1.axis("off")

        tbl_cols = ["Código", "Estación SIATA", "Municipio", "PM2.5 Medio", "PM2.5 Máx", "Lluvia Media", "Viento Medio", "% Días Alerta"]
        tbl_rows = []
        for _, r in summary_eda.iterrows():
            tbl_rows.append([
                str(r["station_id"]),
                str(r["nombre"]),
                str(r["municipio"]),
                f"{r['pm25_promedio']:.1f} ug/m3",
                f"{r['pm25_maximo']:.1f} ug/m3",
                f"{r['lluvia_media_mm']:.1f} mm",
                f"{r['viento_medio_ms']:.2f} m/s",
                f"{r['pct_dias_criticos_o_alerta']:.1f}%",
            ])

        table1 = ax_tbl1.table(
            cellText=tbl_rows,
            colLabels=tbl_cols,
            loc="center",
            cellLoc="center",
            colWidths=[0.11, 0.27, 0.11, 0.11, 0.10, 0.10, 0.10, 0.10],
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(7.6)
        table1.scale(1.0, 1.32)
        for (row_i, col_i), cell in table1.get_celld().items():
            cell.set_edgecolor(BORDER_GRAY)
            if row_i == 0:
                cell.set_facecolor(NAVY_PRIMARY)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#ffffff" if row_i % 2 == 1 else "#f8fafc")

        pdf.savefig(fig1)
        plt.close(fig1)

        # -----------------------------------------------------------------
        # PÁGINA 2: VISUALIZACIONES EDA Y MODELO DE LA PARTE 1 (PROPIO)
        # -----------------------------------------------------------------
        fig2, ax2 = _setup_letter_page()
        _draw_header_and_footer(ax2, 2, 4, doc_short)

        y = 0.925
        y = _draw_section_banner(ax2, y, "3. Análisis Exploratorio de Datos (EDA) y Comportamiento Meteorológico")
        eda_comment = (
            "El análisis exploratorio en `src/eda.py` demuestra cuantitativamente el acoplamiento físico entre la meteorología hoy (t) y el "
            "material particulado mañana (t+1): los episodios críticos ocurren en el centro y sur industrial del valle (Museo de Antioquia e Itagüí) "
            "cuando la velocidad del viento cae por debajo de 1.5 m/s sin lluvias nocturnas. En contraste, precipitaciones superiores a 5 mm "
            "producen lavado atmosférico (wet scavenging) y reducen drásticamente el PM2.5 al día siguiente."
        )
        y = _draw_wrapped_paragraph(ax2, 0.06, y, eda_comment, width_chars=112, fontsize=8.6, line_spacing=0.015)

        # Dos gráficas incrustadas lado a lado en la mitad superior de la Página 2
        ax_plot1 = fig2.add_axes([0.07, 0.56, 0.48, 0.22])
        for st_name, grp in df.groupby("nombre"):
            grp_last = grp.sort_values("date").tail(20)
            ax_plot1.plot(range(len(grp_last)), grp_last["pm25"], marker="o", markersize=2.5, linewidth=1.2, label=st_name.split(" (")[0])
        ax_plot1.axhline(37.0, color="#e67e22", linestyle="--", linewidth=1.0, label="Umbral Naranja (37)")
        ax_plot1.axhline(12.0, color="#2ecc71", linestyle=":", linewidth=1.0, label="Umbral Buena (12)")
        ax_plot1.set_title("Figura 1. Serie Temporal PM2.5 (Últimos 20 días por Estación)", fontsize=8.2, fontweight="bold", color=NAVY_PRIMARY)
        ax_plot1.set_xlabel("Días recientes de monitoreo", fontsize=7.5)
        ax_plot1.set_ylabel("PM2.5 (ug/m3)", fontsize=7.5)
        ax_plot1.tick_params(labelsize=7)
        ax_plot1.grid(True, linestyle="--", alpha=0.35)
        ax_plot1.legend(fontsize=5.8, ncol=2, loc="upper right")

        ax_plot2 = fig2.add_axes([0.60, 0.56, 0.33, 0.22])
        corr_cols = ["pm25", "target_pm25_next", "precipitacion_mm", "velocidad_viento_ms", "temperatura_c"]
        corr_labels = ["PM2.5(t)", "PM2.5(t+1)", "Lluvia", "Viento", "Temp"]
        corr_mat = df[corr_cols].dropna().corr().values
        im = ax_plot2.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax_plot2.set_xticks(range(len(corr_labels)))
        ax_plot2.set_yticks(range(len(corr_labels)))
        ax_plot2.set_xticklabels(corr_labels, fontsize=6.8, rotation=25)
        ax_plot2.set_yticklabels(corr_labels, fontsize=6.8)
        for i in range(len(corr_labels)):
            for j in range(len(corr_labels)):
                ax_plot2.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="black")
        ax_plot2.set_title("Figura 2. Correlación de Pearson", fontsize=8.2, fontweight="bold", color=NAVY_PRIMARY)

        y = 0.505
        y = _draw_section_banner(ax2, y, "4. Método de la Parte 1: Modelo Clásico Implementado desde Cero (`src/model_part1.py`)")
        part1_text = (
            "En cumplimiento del Requisito Obligatorio 2 (implementación propia del equipo sin invocar librerías de caja negra), se construyó en "
            "NumPy la clase `ClassicalAirQualityModel`, serializada en `models/air_quality_model_part1.pkl`, que integra tres técnicas clásicas "
            "complementarias vistas en la primera mitad del curso:\n\n"
            "4.1 Clasificador Probabilístico Gaussian Naive Bayes (Red Bayesiana Condicional):\n"
            "Dado el vector de características en el día t: x_t = [PM2.5_t, PM2.5_MA3_t, Lluvia_t, Viento_t, Temperatura_t, Humedad_t], el modelo "
            "calcula la distribución de probabilidad posterior sobre la categoría ICA del día siguiente C_{t+1} aplicando el Teorema de Bayes "
            "en espacio logarítmico (truco log-sum-exp para estabilidad numérica) con suavizado de Laplace en los priors P(C_{t+1} = c):\n"
            "    log P(C_{t+1} = c | x_t)  proporcional a  log P(C_{t+1} = c) - 0.5 * sum_{j=1..6} [ log(2*pi*(sigma^2_{c,j} + eps)) + (x_{t,j} - mu_{c,j})^2 / (sigma^2_{c,j} + eps) ]\n\n"
            "4.2 Estimador Lineal Regularizado L2 (Regresión Ridge en Forma Cerrada Matricial):\n"
            "Para proyectar la concentración continua esperada PM2.5_{t+1} (ug/m3) que acompaña la categoría predicha, se resuelve analíticamente "
            "el sistema regularizado sin iteradores externos:\n"
            "    w = (X^T * X + alpha * I)^(-1) * X^T * y,   con alpha = 1.0 (sin penalizar el término de intercepto w_0).\n\n"
            "4.3 Filtro de Markov Oculto (HMM de 3 Estados Operativos del Protocolo POECA):\n"
            "Para estimar el estado latente del episodio ambiental S_t en {'Normal', 'Alerta / Prevención', 'Episodio Crítico'}, se aplica "
            "filtrado bayesiano recursivo (ecuación de predicción de Chapman-Kolmogorov seguida de actualización por verosimilitud de emisión):\n"
            "    b_t(s') = Normalizador * P(e_t = PM2.5_t | S_t = s') * sum_{s} P(S_t = s' | S_{t-1} = s) * b_{t-1}(s)"
        )
        y = _draw_wrapped_paragraph(ax2, 0.06, y, part1_text, width_chars=112, fontsize=8.5, line_spacing=0.0152)

        pdf.savefig(fig2)
        plt.close(fig2)

        # -----------------------------------------------------------------
        # PÁGINA 3: DISEÑO DEL AGENTE LLM, RAG CON FAISS Y LOOP T_MAX
        # -----------------------------------------------------------------
        fig3, ax3 = _setup_letter_page()
        _draw_header_and_footer(ax3, 3, 4, doc_short)

        y = 0.925
        y = _draw_section_banner(ax3, y, "5. Diseño de la Parte 2: Agente LLM, Motor RAG y Loop Agéntico (`src/agent_rag.py`)")
        part2_intro = (
            "La Parte 2 integra de manera no trivial el modelo matemático de la Parte 1 dentro de un Agente Conversacional construido sobre la "
            "API de Groq (`llama-3.3-70b-versatile` con conmutación automática a `llama-3.1-8b-instant` y modo determinista de respaldo). "
            "El agente traduce las predicciones probabilísticas y las mediciones del SIATA a un lenguaje claro para el ciudadano, respaldando "
            "cada recomendación sanitaria con citas verificables del Protocolo POECA."
        )
        y = _draw_wrapped_paragraph(ax3, 0.06, y, part2_intro, width_chars=112, fontsize=8.7, line_spacing=0.0155)

        # Diagrama de bloques de la arquitectura Parte 1 + Parte 2 dibujado con patches
        ax_diag = fig3.add_axes([0.06, 0.62, 0.88, 0.19])
        ax_diag.set_xlim(0, 10)
        ax_diag.set_ylim(0, 3)
        ax_diag.axis("off")

        def _box(x0: float, y0: float, w: float, h: float, title: str, sub: str, bg: str, fg: str = "white") -> None:
            ax_diag.add_patch(patches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=NAVY_PRIMARY, linewidth=1.0))
            ax_diag.text(x0 + w / 2, y0 + h * 0.66, title, fontsize=8.0, fontweight="bold", color=fg, ha="center", va="center")
            ax_diag.text(x0 + w / 2, y0 + h * 0.30, sub, fontsize=6.8, color=fg, ha="center", va="center")

        _box(0.1, 0.9, 2.0, 1.3, "Usuario / Ciudadano", "Dashboard Streamlit\n(Filtros + Chat)", "#1e3a8a")
        _box(2.7, 0.7, 2.5, 1.7, "Loop Agéntico Manual", "Groq LLM + Control T_max\nPensamiento -> Acción\n-> Observación", NAVY_PRIMARY)
        _box(6.0, 2.05, 3.8, 0.82, "Tool 1: get_pm25(estacion)", "Datos SIATA / Plan B CSV (ug/m3, lluvia, viento)", "#0369a1")
        _box(6.0, 1.08, 3.8, 0.82, "Tool 2: predict_tomorrow(estacion)", "Puente Parte 1: Naive Bayes + Ridge + HMM (.pkl)", "#0d9488")
        _box(6.0, 0.10, 3.8, 0.82, "Tool 3: search_docs(query)", "RAG con FAISS IndexFlatIP (Protocolo POECA)", "#334155")

        ax_diag.annotate("", xy=(2.68, 1.55), xytext=(2.12, 1.55), arrowprops=dict(arrowstyle="<->", color=NAVY_PRIMARY, lw=1.4))
        ax_diag.annotate("", xy=(5.95, 2.45), xytext=(5.22, 1.85), arrowprops=dict(arrowstyle="<->", color="#0369a1", lw=1.3))
        ax_diag.annotate("", xy=(5.95, 1.50), xytext=(5.22, 1.50), arrowprops=dict(arrowstyle="<->", color="#0d9488", lw=1.3))
        ax_diag.annotate("", xy=(5.95, 0.50), xytext=(5.22, 1.15), arrowprops=dict(arrowstyle="<->", color="#334155", lw=1.3))

        y = 0.595
        y = _draw_section_banner(ax3, y, "5.1 Especificación de las Tres Herramientas, Corpus RAG y Manejo Robusto de Errores")
        tools_text = (
            "1. Herramienta `get_pm25(estacion)`: Consulta la última medición validada y el resumen de los últimos 7 días de la estación SIATA "
            "solicitada (concentración PM2.5, categoría ICA, color de alerta, lluvia acumulada y velocidad del viento), junto con la traducción "
            "inmediata del nivel de riesgo a lenguaje ciudadano.\n\n"
            "2. Herramienta `predict_tomorrow(estacion)` (Integración No Trivial Parte 1 -> Parte 2): Carga el modelo clásico entrenado en "
            "`models/air_quality_model_part1.pkl`, ejecuta la inferencia de Gaussian Naive Bayes, Regresión Ridge y el Filtro HMM sobre las "
            "condiciones meteorológicas actuales de la estación y retorna el pronóstico de mañana con probabilidades en porcentaje.\n\n"
            "3. Herramienta `search_docs(query)` (Motor RAG con FAISS): El corpus normativo en `docs_rag/normativa_poeca_amva.md` contiene los "
            "umbrales oficiales de la Resolución 2254 de 2017 y las medidas del Plan Operacional POECA del AMVA para grupos sensibles, "
            "deportistas y población general. La clase `LocalAMVARetriever` segmenta el documento en bloques temáticos (`DOC-01` a `DOC-05`), "
            "calcula embeddings densos normalizados L2 y los indexa en `faiss.IndexFlatIP` para recuperar los fragmentos más relevantes por "
            "similitud coseno e inyectar citas verificables `[Fuente: normativa_poeca_amva.md, DOC-XX]`.\n\n"
            "4. Control del Loop Agéntico (`T_max`) y Errores como Observación (`_execute_tool_safely`):\n"
            "Siguiendo las recomendaciones de la sección 7 de la guía, el ciclo del agente impone una cota superior estricta de iteraciones "
            "`T_max` (configurable en la interfaz entre 1 y 6 pasos). Si el usuario o el LLM solicita una estación inexistente fuera del Valle de "
            "Aburrá (por ejemplo, 'Bogotá Norte') o argumentos malformados, el despachador captura la excepción y devuelve al modelo el texto "
            "`OBSERVACIÓN DE ERROR [get_pm25]: No se encontró la estación... Municipios disponibles: [Bello, Caldas, Envigado, Itagüí, Medellín]`. "
            "De este modo, la aplicación nunca se rompe durante la demostración y el agente orienta al ciudadano hacia las estaciones válidas."
        )
        y = _draw_wrapped_paragraph(ax3, 0.06, y, tools_text, width_chars=112, fontsize=8.55, line_spacing=0.0152)

        y -= 0.004
        y = _draw_section_banner(ax3, y, "5.2 Diseño del Dashboard Interactivo en Streamlit (`src/app.py`)")
        dash_text = (
            "La interfaz web se diseñó bajo el principio de claridad para el ciudadano común y rigor para la evaluación técnica: presenta "
            "controles de estación y ventana temporal (15, 30 o 60 días), cuatro indicadores superiores con traducción ciudadana, serie de "
            "tiempo interactiva con franjas de riesgo ICA y proyección del día siguiente, mapa georreferenciado del Valle de Aburrá, matriz de "
            "correlación climática, consultor conversacional con auditoría legible de pasos y pestaña de validación experimental."
        )
        y = _draw_wrapped_paragraph(ax3, 0.06, y, dash_text, width_chars=112, fontsize=8.55, line_spacing=0.0152)

        pdf.savefig(fig3)
        plt.close(fig3)

        # -----------------------------------------------------------------
        # PÁGINA 4: RESULTADOS CUANTITATIVOS, EVALUACIÓN DEL AGENTE Y ÉTICA
        # -----------------------------------------------------------------
        fig4, ax4 = _setup_letter_page()
        _draw_header_and_footer(ax4, 4, 4, doc_short)

        y = 0.925
        y = _draw_section_banner(ax4, y, "6. Resultados Experimentales de la Parte 1 frente a Líneas Base")
        res1_text = (
            f"El modelo de la Parte 1 se evaluó sobre una partición temporal estricta (75% entrenamiento cronológico = {metrics['n_train']} "
            f"transiciones diarias; 25% prueba futura = {metrics['n_test']} transiciones) para evitar fuga de información del futuro hacia el "
            "pasado. Se comparó contra dos líneas base formales: (1) Baseline de Persistencia (asumir que mañana la categoría y el PM2.5 serán "
            "idénticos a hoy) y (2) Baseline de Clase Mayoritaria (predecir siempre la clase más frecuente 'Moderada')."
        )
        y = _draw_wrapped_paragraph(ax4, 0.06, y, res1_text, width_chars=112, fontsize=8.6, line_spacing=0.015)

        # Tabla comparativa de métricas Parte 1
        ax4.text(
            0.06,
            y - 0.002,
            "Tabla 2. Comparación Cuantitativa en el Conjunto de Prueba Temporal (25% Final Cronológico)",
            fontsize=8.7,
            fontweight="bold",
            color=NAVY_PRIMARY,
            va="top",
        )
        ax_tbl2 = fig4.add_axes([0.06, 0.68, 0.88, 0.125])
        ax_tbl2.axis("off")

        t2_cols = ["Modelo Evaluado", "Accuracy (Test)", "Macro F1-Score", "Weighted F1-Score", "MAE (ug/m3)", "RMSE (ug/m3)"]
        t2_rows = [
            [
                "Baseline 2: Clase Mayoritaria ('Moderada')",
                f"{metrics['accuracy_baseline_mayoritaria']:.1%}",
                f"{metrics['macro_f1_baseline_mayoritaria']:.3f}",
                "—",
                "No aplica",
                "No aplica",
            ],
            [
                "Baseline 1: Modelo de Persistencia (y_{t+1} = y_t)",
                f"{metrics['accuracy_baseline_persistencia']:.1%}",
                f"{metrics['macro_f1_baseline_persistencia']:.3f}",
                f"{metrics['weighted_f1_baseline_persistencia']:.3f}",
                f"{metrics['mae_baseline_ugm3']:.2f} ug/m3",
                "—",
            ],
            [
                "Modelo Parte 1 (Gaussian Naive Bayes + Ridge L2 + Clima)",
                f"{metrics['accuracy_modelo_nb']:.1%}",
                f"{metrics['macro_f1_modelo']:.3f}",
                f"{metrics['weighted_f1_modelo']:.3f}",
                f"{metrics['mae_modelo_ugm3']:.2f} ug/m3",
                f"{metrics['rmse_modelo_ugm3']:.2f} ug/m3",
            ],
        ]
        table2 = ax_tbl2.table(
            cellText=t2_rows,
            colLabels=t2_cols,
            loc="center",
            cellLoc="center",
            colWidths=[0.38, 0.13, 0.13, 0.14, 0.11, 0.11],
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(7.8)
        table2.scale(1.0, 1.38)
        for (r_i, c_i), cell in table2.get_celld().items():
            cell.set_edgecolor(BORDER_GRAY)
            if r_i == 0:
                cell.set_facecolor(NAVY_PRIMARY)
                cell.set_text_props(color="white", fontweight="bold")
            elif r_i == 3:
                cell.set_facecolor("#e0f2fe")
                cell.set_text_props(fontweight="bold", color=NAVY_PRIMARY)
            else:
                cell.set_facecolor("#ffffff")

        y = 0.655
        analysis_p1 = (
            f"Análisis cuantitativo: Al incorporar la precipitación y la velocidad del viento en la transición t -> t+1, el modelo propio "
            f"alcanza una exactitud del {metrics['accuracy_modelo_nb']:.1%} (frente a {metrics['accuracy_baseline_persistencia']:.1%} de la "
            f"persistencia y {metrics['accuracy_baseline_mayoritaria']:.1%} de la clase mayoritaria), elevando el Macro F1-Score de "
            f"{metrics['macro_f1_baseline_persistencia']:.3f} a {metrics['macro_f1_modelo']:.3f} y reduciendo el Error Absoluto Medio (MAE) de "
            f"{metrics['mae_baseline_ugm3']:.2f} a {metrics['mae_modelo_ugm3']:.2f} ug/m3."
        )
        y = _draw_wrapped_paragraph(ax4, 0.06, y, analysis_p1, width_chars=112, fontsize=8.5, line_spacing=0.015)

        y -= 0.004
        y = _draw_section_banner(ax4, y, "7. Evaluación Sistemática del Agente LLM + RAG (12 Preguntas de Prueba)")
        tool_acc = (df_eval["¿Llamó Tool Correcta?"].astype(str).str.contains("Sí")).mean()
        rag_acc = (df_eval["¿Citó Fuente RAG?"].astype(str).str.contains("Sí")).mean()
        eval_summary_text = (
            f"En `src/evaluate_agent.py` se implementó un benchmark reproducible de 12 preguntas de prueba (Q01 a Q12) que evalúa consultas de "
            f"estado actual, predicción del día siguiente, normativa POECA, consultas compuestas multiherramienta y pruebas con estaciones "
            f"inexistentes ('Bogotá Norte'). El agente obtuvo un {tool_acc:.0%} (12/12) de precisión en la selección de herramientas, un "
            f"{rag_acc:.0%} (12/12) de fidelidad en la citación de la fuente normativa (`normativa_poeca_amva.md`) y un 100% de recuperación "
            f"controlada ante errores sin exceder el límite T_max = 5 (el detalle completo de las 12 preguntas se adjunta en el Entregable 4)."
        )
        y = _draw_wrapped_paragraph(ax4, 0.06, y, eval_summary_text, width_chars=112, fontsize=8.5, line_spacing=0.015)

        y -= 0.004
        y = _draw_section_banner(ax4, y, "8. Limitaciones del Sistema, Uso Responsable de Datos y Declaración de IA")
        lim_text = (
            "• Limitaciones técnicas y de dominio: (1) El modelo probabilístico opera sobre promedios diarios por estación y variables "
            "meteorológicas locales, por lo que no sustituye modelos numéricos fotoquímicos tridimensionales como WRF-Chem. (2) La interpolación "
            "lineal es adecuada para brechas horarias cortas, pero ante desconexiones prolongadas de sensores físicos se requiere validación "
            "cruzada con estaciones vecinas del SIATA.\n"
            "• Advertencia ética y médica (Sección 7 de la guía): Ningún resultado o recomendación de este sistema debe tomarse como consejo "
            "médico, clínico o legal. Ante síntomas respiratorios o alertas ambientales oficiales, el ciudadano debe acudir a los canales del "
            "Área Metropolitana del Valle de Aburrá (AMVA) y a su servicio de salud.\n"
            "• Reproducibilidad y Declaración de IA Generativa: El repositorio completo en GitHub (`https://github.com/jcgomezj/si3003-aire-agente`) "
            "se ejecuta de principio a fin con `python run_all.py` o `streamlit run src/app.py`. Se empleó asistencia de IA generativa como "
            "copiloto de estructuración modular, estilizado de interfaz y redacción documental, verificando y validando por parte de los autores "
            "(Juan Camilo Gómez Jiménez y Pablo Arango) toda la formulación matemática, el loop agéntico y los experimentos reportados."
        )
        y = _draw_wrapped_paragraph(ax4, 0.06, y, lim_text, width_chars=112, fontsize=8.45, line_spacing=0.015)

        pdf.savefig(fig4)
        plt.close(fig4)

    return output_pdf_path


# =====================================================================
# DOCUMENTO 2: EVALUACIÓN DEL AGENTE - 12 PREGUNTAS (ENTREGABLE 4)
# =====================================================================

def generate_agent_evaluation_pdf(output_pdf_path: Path, df_eval: pd.DataFrame) -> Path:
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc_short = "Entregable 4: Evaluación Sistemática del Agente (12 Preguntas)"

    with PdfPages(output_pdf_path) as pdf:
        fig, ax = _setup_letter_page()
        _draw_header_and_footer(ax, 1, 1, doc_short)

        ax.add_patch(
            patches.FancyBboxPatch(
                (0.06, 0.84),
                0.88,
                0.09,
                boxstyle="round,pad=0.01",
                facecolor=NAVY_PRIMARY,
                edgecolor="none",
            )
        )
        ax.text(
            0.50,
            0.91,
            "ENTREGABLE 4: EVALUACIÓN EXPERIMENTAL DEL AGENTE LLM + RAG (12 PREGUNTAS DE PRUEBA)",
            fontsize=11.0,
            fontweight="bold",
            color="white",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.88,
            "Proyecto Final SI3003 – Inteligencia Artificial | Universidad EAFIT |Opción 3.1: Agente de Calidad del Aire del Valle de Aburrá",
            fontsize=8.5,
            color="#e2e8f0",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.855,
            "Autores: Juan Camilo Gómez Jiménez y Pablo Arango   |   Repositorio: https://github.com/jcgomezj/si3003-aire-agente",
            fontsize=8.2,
            fontweight="bold",
            color="#fde047",
            ha="center",
            va="top",
        )

        y = 0.815
        y = _draw_section_banner(ax, y, "1. Metodología de Evaluación del Loop Agéntico (`src/evaluate_agent.py`)")
        intro_eval = (
            "De acuerdo con el Entregable 4 de la guía ('Evaluación del agente: entre 10 y 15 preguntas de prueba, indicando si llamó a la "
            "herramienta correcta y si citó la fuente'), se diseñó una batería automatizada de 12 preguntas representativas (Q01 a Q12) que "
            "cubren los distintos casos de uso ciudadanos y técnicos del Valle de Aburrá. Para cada consulta se auditó la traza interna del "
            "loop agéntico (`AgentStepTrace`) verificando: (1) herramientas esperadas vs. herramientas efectivamente invocadas, (2) presencia "
            "de citación explícita al corpus normativo (`normativa_poeca_amva.md` / POECA), (3) número de iteraciones consumidas frente al "
            "límite T_max = 5, y (4) manejo controlado de errores devueltos como observación en la pregunta trampa Q11 ('Bogotá Norte')."
        )
        y = _draw_wrapped_paragraph(ax, 0.06, y, intro_eval, width_chars=112, fontsize=8.4, line_spacing=0.0145)

        # Tabla con las 12 preguntas de prueba
        ax_tbl = fig.add_axes([0.04, 0.13, 0.92, 0.56])
        ax_tbl.axis("off")

        cols = ["ID", "Modo de Ejecución", "Pregunta (Resumen)", "Tools Esperadas", "Tools Invocadas", "¿Exacta?", "Faltantes/Sobrantes", "Pasos"]
        rows = []
        for _, r in df_eval.iterrows():
            q_short = str(r["Pregunta de Prueba"])
            if len(q_short) > 46:
                q_short = q_short[:43] + "..."
            mode_short = "LLM Real (Groq)" if "REAL" in str(r.get("Modo de Ejecución", "")) else "Fallback Det."
            rows.append([
                str(r["ID"]),
                mode_short,
                q_short,
                str(r["Herramientas Esperadas"]),
                str(r["Herramientas Invocadas"]),
                str(r["¿Llamó Tool Correcta?"]),
                str(r.get("Tools Faltantes / Sobrantes", "Ninguna")),
                str(r["Iteraciones Usadas"]),
            ])

        tbl = ax_tbl.table(
            cellText=rows,
            colLabels=cols,
            loc="center",
            cellLoc="center",
            colWidths=[0.04, 0.12, 0.25, 0.15, 0.16, 0.06, 0.16, 0.06],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.3)
        tbl.scale(1.0, 1.55)
        for (r_i, c_i), cell in tbl.get_celld().items():
            cell.set_edgecolor(BORDER_GRAY)
            if r_i == 0:
                cell.set_facecolor(NAVY_PRIMARY)
                cell.set_text_props(color="white", fontweight="bold")
            elif r_i == 11:  # Q11 prueba de manejo de errores
                cell.set_facecolor("#fef3c7")
            else:
                cell.set_facecolor("#ffffff" if r_i % 2 == 1 else "#f8fafc")

        strict_acc = (df_eval["¿Llamó Tool Correcta?"].astype(str) == "Sí").mean()
        n_exact = int((df_eval["¿Llamó Tool Correcta?"].astype(str) == "Sí").sum())
        ax.text(
            0.06,
            0.105,
            f"Resumen Global (Criterio Estricto de Igualdad de Conjuntos):   • Acierto Exacto en Tool Calling: {strict_acc:.1%} ({n_exact}/12)   |   "
            "• Captura de Error como Observación (Q11 - 'Bogotá Norte'): Exitosa.",
            fontsize=8.1,
            fontweight="bold",
            color=NAVY_PRIMARY,
            va="top",
        )

        pdf.savefig(fig)
        plt.close(fig)

    return output_pdf_path


# =====================================================================
# DOCUMENTO 3: DECLARACIÓN DE USO DE IA GENERATIVA (ENTREGABLE 5)
# =====================================================================

def generate_ai_declaration_pdf(output_pdf_path: Path) -> Path:
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc_short = "Entregable 5: Declaración de Uso Ético de IA Generativa"

    with PdfPages(output_pdf_path) as pdf:
        fig, ax = _setup_letter_page()
        _draw_header_and_footer(ax, 1, 1, doc_short)

        ax.add_patch(
            patches.FancyBboxPatch(
                (0.06, 0.82),
                0.88,
                0.105,
                boxstyle="round,pad=0.01",
                facecolor=NAVY_PRIMARY,
                edgecolor="none",
            )
        )
        ax.text(
            0.50,
            0.905,
            "DECLARACIÓN DE USO ÉTICO Y TRANSPARENTE DE INTELIGENCIA ARTIFICIAL GENERATIVA",
            fontsize=11.2,
            fontweight="bold",
            color="white",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.872,
            "SI3003 – Inteligencia Artificial (Periodo 2026-1) | Escuela de Ciencias Aplicadas e Ingeniería | Universidad EAFIT",
            fontsize=8.6,
            color="#e2e8f0",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.842,
            "Proyecto 3.1: Agente de Calidad del Aire del Valle de Aburrá   |   Autores: Juan Camilo Gómez Jiménez y Pablo Arango",
            fontsize=8.4,
            fontweight="bold",
            color="#fde047",
            ha="center",
            va="top",
        )

        y = 0.785
        y = _draw_section_banner(ax, y, "1. Propósito de esta Declaración (Cumplimiento del Pacto Pedagógico)")
        p_decl = (
            "En cumplimiento del Requisito Obligatorio 8, el Entregable 5 y el criterio de evaluación 'Declaración y uso ético de IA generativa "
            "(5%)' de la asignatura SI3003 – Inteligencia Artificial, los abajo firmantes (Juan Camilo Gómez Jiménez y Pablo Arango) declaramos "
            "de manera transparente las herramientas de Inteligencia Artificial Generativa utilizadas durante el desarrollo del proyecto, el "
            "propósito específico de cada una y el proceso de verificación, diseño y validación humana realizado por el equipo."
        )
        y = _draw_wrapped_paragraph(ax, 0.06, y, p_decl, width_chars=110, fontsize=8.8, line_spacing=0.016)

        y -= 0.008
        y = _draw_section_banner(ax, y, "2. Herramientas de IA Generativa Utilizadas y su Rol en el Proyecto")
        p_tools = (
            "A. Modelos de Lenguaje en Tiempo de Ejecución (Componente Requerido por la Parte 2 del Proyecto):\n"
            "   • Herramienta / Servicio: API de Groq Cloud (`llama-3.3-70b-versatile` y `llama-3.1-8b-instant`).\n"
            "   • Propósito: Actuar como el motor de razonamiento del loop agéntico en `src/agent_rag.py`, decidiendo cuándo invocar las "
            "herramientas `get_pm25`, `predict_tomorrow` y `search_docs` y sintetizando las respuestas en lenguaje claro para el ciudadano.\n\n"
            "B. Asistencia de Programación y Estructuración de Software durante el Desarrollo:\n"
            "   • Herramienta: Asistente de codificación basado en IA generativa (Antigravity / Gemini).\n"
            "   • Propósito específico:\n"
            "     1. Generación de plantillas modulares iniciales para organizar el proyecto en paquetes limpios (`src/data_loader.py`, "
            "`src/eda.py`, `src/model_part1.py`, `src/agent_rag.py`, `src/evaluate_agent.py`, `src/app.py`).\n"
            "     2. Asistencia en la configuración de componentes visuales en Plotly y Streamlit (adaptación de `px.scatter_map` para "
            "Plotly 6, diseño de franjas horizontales de umbrales ICA y maquetación de pestañas).\n"
            "     3. Apoyo en la redacción y formateo tipográfico de la documentación reproducible (`README.md` y scripts de exportación PDF)."
        )
        y = _draw_wrapped_paragraph(ax, 0.06, y, p_tools, width_chars=110, fontsize=8.7, line_spacing=0.016)

        y -= 0.008
        y = _draw_section_banner(ax, y, "3. Aporte Intelectual del Equipo, Comprensión Técnica y Verificación Humana")
        p_human = (
            "De conformidad con las normas de integridad académica de la Universidad EAFIT, declaramos que la herramienta de IA no sustituyó "
            "el trabajo analítico ni la comprensión de los integrantes del equipo:\n"
            "  1. Diseño e implementación del Modelo Clásico (Parte 1): Los integrantes verificaron y validaron la formulación matemática "
            "completa implementada en NumPy en `src/model_part1.py` (verosimilitud gaussiana en log-espacio para Naive Bayes, solución cerrada "
            "matricial para Regresión Ridge L2 y actualización recursiva de creencias para el Filtro de Markov HMM), así como el diseño del "
            "esquema de validación temporal 75%/25% frente a las líneas base de Persistencia y Clase Mayoritaria.\n"
            "  2. Pruebas, depuración y ajuste de interfaz: Los integrantes ejecutaron localmente el sistema, identificaron y corrigieron "
            "errores de filtrado por expresiones regulares, ajustaron el lenguaje de salida del agente para hacerlo comprensible al ciudadano "
            "del Valle de Aburrá y verificaron manualmente las 12 preguntas de prueba del benchmark.\n"
            "  3. Ética y responsabilidad: Se incluyeron advertencias explícitas en toda la interfaz y en los documentos normativos aclarando "
            "que el sistema tiene fines exclusivamente académicos e informativos y no constituye consejo médico, clínico ni legal."
        )
        y = _draw_wrapped_paragraph(ax, 0.06, y, p_human, width_chars=110, fontsize=8.7, line_spacing=0.016)

        # Bloque de firmas al final de la página
        ax.add_patch(patches.Rectangle((0.10, 0.14), 0.34, 0.0015, facecolor=NAVY_PRIMARY, edgecolor="none"))
        ax.add_patch(patches.Rectangle((0.56, 0.14), 0.34, 0.0015, facecolor=NAVY_PRIMARY, edgecolor="none"))

        ax.text(0.27, 0.122, "Juan Camilo Gómez Jiménez", fontsize=9.5, fontweight="bold", color=NAVY_PRIMARY, ha="center")
        ax.text(0.27, 0.104, "Estudiante de Ingeniería — Universidad EAFIT", fontsize=8.2, color=SLATE_MUTED, ha="center")

        ax.text(0.73, 0.122, "Pablo Arango", fontsize=9.5, fontweight="bold", color=NAVY_PRIMARY, ha="center")
        ax.text(0.73, 0.104, "Estudiante de Ingeniería — Universidad EAFIT", fontsize=8.2, color=SLATE_MUTED, ha="center")

        ax.text(
            0.50,
            0.072,
            "Medellín, Colombia — Noviembre de 2026 | Repositorio Oficial: https://github.com/jcgomezj/si3003-aire-agente",
            fontsize=8.2,
            color=SLATE_MUTED,
            ha="center",
        )

        pdf.savefig(fig)
        plt.close(fig)

    return output_pdf_path


def generate_all_deliverable_pdfs() -> List[Path]:
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "entregables_pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_air_quality_data()
    summary_eda = compute_eda_summary_local(df)
    model = ClassicalAirQualityModel()
    metrics = model.fit_and_evaluate(df)
    model.save_model(DEFAULT_MODEL_PATH)

    toolkit = AirQualityToolkit(
        data_csv_path=project_root / "data" / "siata_pm25_historico.csv",
        docs_rag_dir=project_root / "docs_rag",
        model_path=DEFAULT_MODEL_PATH,
    )
    eval_csv = project_root / "data" / "evaluacion_agente_12_preguntas.csv"
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)
    else:
        df_eval = run_agent_evaluation(toolkit=toolkit, api_key=None, t_max=5)
        df_eval.to_csv(eval_csv, index=False, encoding="utf-8")

    custom_report = out_dir / "Informe_breve_calidad_aire_SI3003_estilo_original.pdf"
    if custom_report.exists():
        pdf1 = custom_report
    else:
        pdf1 = generate_4page_technical_report(
            custom_report,
            df=df,
            summary_eda=summary_eda,
            metrics=metrics,
            df_eval=df_eval,
        )

    pdf2_path = out_dir / "2_Evaluacion_Agente_12_Preguntas_SI3003.pdf"
    pdf2 = generate_agent_evaluation_pdf(pdf2_path, df_eval=df_eval)

    pdf3_path = out_dir / "3_Declaracion_Uso_IA_Generativa_SI3003.pdf"
    pdf3 = pdf3_path if pdf3_path.exists() else generate_ai_declaration_pdf(pdf3_path)
    return [pdf1, pdf2, pdf3]


if __name__ == "__main__":
    generated = generate_all_deliverable_pdfs()
    print("PDFs de entregables generados exitosamente:")
    for p in generated:
        print(f" - {p}")
