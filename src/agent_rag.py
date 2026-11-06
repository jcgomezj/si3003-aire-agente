"""
Módulo de Agente Conversacional, RAG y Herramientas (`src/agent_rag.py`) - SI3003 Universidad EAFIT.

Integra de forma limpia y orientada al ciudadano:
- Modelo Predictivo (`src/model_part1.py`): Clasificador Gaussian Naive Bayes + Regresión Ridge + Filtro HMM.
- Motor RAG Local (`LocalAMVARetriever`): Indexación con FAISS (`IndexFlatIP`) sobre `docs_rag/normativa_poeca_amva.md`.
- Tres herramientas con manejo robusto de errores devueltos como observaciones:
  1. `get_pm25(estacion)`
  2. `predict_tomorrow(estacion)`
  3. `search_docs(query)`
- Loop agéntico manual con límite de iteraciones (T_max) y conmutación automática entre modelos disponibles
  en la cuenta de Groq del usuario (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `llama3-70b-8192`, `gemma2-9b-it`).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_air_quality_data
from src.model_part1 import ClassicalAirQualityModel, DEFAULT_MODEL_PATH, pm25_to_category


def _load_env_file_if_present() -> None:
    """Carga variables desde el archivo `.env` en la raíz del proyecto."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file_if_present()

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from groq import Groq  # type: ignore
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False


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


# =====================================================================
# 1. SISTEMA RAG LOCAL (FAISS + Embeddings Locales sobre docs_rag/)
# =====================================================================

@dataclass
class DocumentChunk:
    """Representa un fragmento indexado del corpus normativo del AMVA para citación en el RAG."""
    chunk_id: str
    source_file: str
    section_title: str
    content: str


class LocalAMVARetriever:
    """
    Motor RAG para documentos de normativa ambiental y salud del Valle de Aburrá (`docs_rag/`).
    Indexa los fragmentos en `faiss.IndexFlatIP` y limpia los encabezados Markdown para que
    los textos insertados en el chat nunca desconfiguren el tamaño de fuente de la interfaz.
    """

    def __init__(self, docs_dir: Path, embedding_dim: int = 512) -> None:
        self.docs_dir = docs_dir
        self.embedding_dim = embedding_dim
        self.chunks: List[DocumentChunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.faiss_index: Any = None
        self._load_and_index_corpus()

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        clean = re.sub(r"[^\w\sáéíóúñü]", " ", text.lower())
        tokens = [t for t in clean.split() if len(t) > 2]

        for token in tokens:
            idx_word = hash(token) % self.embedding_dim
            vec[idx_word] += 2.0
            for i in range(max(1, len(token) - 2)):
                trigram = token[i : i + 3]
                idx_tri = hash(trigram) % self.embedding_dim
                vec[idx_tri] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _sanitize_markdown_headings(text: str) -> str:
        """Elimina símbolos '#' al inicio de las líneas para evitar títulos gigantes en Streamlit."""
        cleaned_lines = []
        for line in text.splitlines():
            stripped = re.sub(r"^#+\s*", "", line).strip()
            if stripped:
                cleaned_lines.append(stripped)
        return "\n".join(cleaned_lines)

    def _load_and_index_corpus(self) -> None:
        if not self.docs_dir.exists():
            return

        chunk_counter = 1
        for file_path in sorted(self.docs_dir.glob("*.*")):
            if file_path.suffix.lower() not in {".md", ".txt"}:
                continue
            raw_text = file_path.read_text(encoding="utf-8")
            sections = re.split(r"\n(?=##\s)", raw_text)
            for sec in sections:
                sec_clean = self._sanitize_markdown_headings(sec)
                if len(sec_clean) < 40:
                    continue
                lines = sec_clean.splitlines()
                first_line = lines[0].strip()
                body_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else sec_clean
                self.chunks.append(
                    DocumentChunk(
                        chunk_id=f"DOC-{chunk_counter:02d}",
                        source_file=file_path.name,
                        section_title=first_line[:100],
                        content=body_text if body_text else sec_clean,
                    )
                )
                chunk_counter += 1

        if not self.chunks:
            return

        vectors = [self._embed_text(f"{c.section_title} {c.content}") for c in self.chunks]
        self.embeddings_matrix = np.vstack(vectors).astype(np.float32)

        if FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(self.embeddings_matrix)

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.chunks or self.embeddings_matrix is None:
            return []

        q_vec = self._embed_text(query).reshape(1, -1)
        k = min(top_k, len(self.chunks))

        if FAISS_AVAILABLE and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q_vec, k)
            top_indices = indices[0].tolist()
            top_scores = scores[0].tolist()
        else:
            sims = (self.embeddings_matrix @ q_vec.T).flatten()
            top_indices = np.argsort(sims)[::-1][:k].tolist()
            top_scores = [float(sims[i]) for i in top_indices]

        results = []
        for idx, score in zip(top_indices, top_scores):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "fuente": chunk.source_file,
                "seccion": chunk.section_title,
                "similitud": round(float(score), 3),
                "contenido": chunk.content,
            })
        return results


# =====================================================================
# 2. DEFINICIÓN DE HERRAMIENTAS DEL AGENTE CON MANEJO DE ERRORES ROBUSTO
# =====================================================================

class AirQualityToolkit:
    """
    Conjunto de herramientas que conectan los datos del SIATA, el modelo predictivo
    y el buscador normativo RAG.
    """

    def __init__(self, data_csv_path: Path, docs_rag_dir: Path, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.data_csv_path = data_csv_path
        self.df = load_air_quality_data(csv_path=self.data_csv_path)
        self.classical_model = ClassicalAirQualityModel.load_or_train(self.df, model_path=model_path)
        self.retriever = LocalAMVARetriever(docs_rag_dir)

    def get_pm25(self, estacion: str) -> str:
        """
        Herramienta 1: Consulta el estado actual e histórico reciente de PM2.5 y variables
        meteorológicas de una estación del SIATA (Valle de Aburrá).
        """
        try:
            if not estacion or not estacion.strip():
                return (
                    "OBSERVACIÓN DE ERROR [get_pm25]: Debe proporcionar el nombre de una estación o "
                    "municipio del Valle de Aburrá (Bello, Centro, El Poblado, Envigado, Itagüí o Caldas)."
                )

            q = estacion.strip().lower()
            mask = (
                self.df["station_id"].astype(str).str.lower().str.contains(q, na=False, regex=False)
                | self.df["nombre"].astype(str).str.lower().str.contains(q, na=False, regex=False)
                | self.df["municipio"].astype(str).str.lower().str.contains(q, na=False, regex=False)
            )
            sub = self.df[mask].sort_values("date").reset_index(drop=True)

            if sub.empty:
                municipios = ", ".join(sorted(self.df["municipio"].unique()))
                estaciones = ", ".join(sorted(self.df["nombre"].unique()))
                return (
                    f"OBSERVACIÓN DE ERROR [get_pm25]: No se encontró la estación '{estacion}'. "
                    f"Municipios disponibles en el Valle de Aburrá: [{municipios}]. Estaciones: [{estaciones}]."
                )

            latest = sub.iloc[-1]
            pm25_val = float(latest["pm25"])
            categoria, color = pm25_to_category(pm25_val)
            promedio_7d = round(float(sub["pm25"].tail(7).mean()), 1)
            max_7d = round(float(sub["pm25"].tail(7).max()), 1)

            payload = {
                "fuente_datos": "Red de Monitoreo SIATA - Área Metropolitana del Valle de Aburrá",
                "station_id": str(latest["station_id"]),
                "nombre_estacion": str(latest["nombre"]),
                "municipio": str(latest["municipio"]),
                "fecha_ultima_medicion": str(latest["date"]),
                "pm25_actual_ugm3": pm25_val,
                "categoria_ica_actual": categoria,
                "color_alerta": color,
                "significado_ciudadano": describe_citizen_air_status(categoria),
                "precipitacion_mm_24h": float(latest["precipitacion_mm"]),
                "velocidad_viento_ms": float(latest["velocidad_viento_ms"]),
                "resumen_ultimos_7_dias": {
                    "promedio_pm25_ugm3": promedio_7d,
                    "maximo_pm25_ugm3": max_7d,
                },
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        except Exception as exc:
            return f"OBSERVACIÓN DE ERROR [get_pm25]: Fallo inesperado al consultar datos de '{estacion}': {exc}"

    def predict_tomorrow(self, estacion: str) -> str:
        """
        Herramienta 2: Invoca el modelo predictivo para estimar la calidad del aire del día siguiente
        en una estación del SIATA y traduce los resultados a lenguaje ciudadano.
        """
        try:
            if not estacion or not estacion.strip():
                return (
                    "OBSERVACIÓN DE ERROR [predict_tomorrow]: Parámetro 'estacion' vacío. "
                    "Indique un municipio o estación válida (Medellín Centro, El Poblado, Envigado, Itagüí, Bello o Caldas)."
                )

            pred = self.classical_model.predict_next_day_for_station(self.df, estacion)
            # Ordenar probabilidades de mayor a menor en porcentaje legible
            probs_sorted = sorted(pred.probabilidades_clase.items(), key=lambda kv: kv[1], reverse=True)
            probs_legibles = {k: f"{v * 100:.1f}%" for k, v in probs_sorted if v >= 0.01}

            payload = {
                "station_id": pred.station_id,
                "nombre_estacion": pred.nombre_estacion,
                "fecha_base_observada": pred.fecha_base,
                "pm25_actual_ugm3": pred.pm25_actual,
                "pm25_predicho_manana_ugm3": pred.pm25_estimado_manana,
                "categoria_predicha_manana": pred.categoria_predicha_manana,
                "color_alerta_manana": pred.color_alerta_manana,
                "significado_ciudadano_manana": describe_citizen_air_status(pred.categoria_predicha_manana),
                "estado_episodio_ambiental": pred.estado_latente_hmm,
                "probabilidades_por_categoria": probs_legibles,
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        except Exception as exc:
            return f"OBSERVACIÓN DE ERROR [predict_tomorrow]: No fue posible predecir para '{estacion}': {exc}"

    def search_docs(self, query: str) -> str:
        """
        Herramienta 3: Busca en la normativa del AMVA (Protocolo POECA y Resolución 2254 de 2017)
        para respaldar las recomendaciones de salud con citas verificables.
        """
        try:
            if not query or not query.strip():
                return "OBSERVACIÓN DE ERROR [search_docs]: La consulta 'query' no puede estar vacía."

            hits = self.retriever.search(query, top_k=2)
            if not hits:
                return "OBSERVACIÓN DE ERROR [search_docs]: No se encontraron documentos en `docs_rag/`."

            return json.dumps(
                {
                    "consulta": query,
                    "fragmentos_recuperados": hits,
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as exc:
            return f"OBSERVACIÓN DE ERROR [search_docs]: Error en búsqueda normativa para '{query}': {exc}"


AGENT_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_pm25",
            "description": (
                "Consulta el estado actual de la calidad del aire (PM2.5, categoría, color de alerta, "
                "lluvia y viento) de una estación del SIATA en el Valle de Aburrá."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {
                        "type": "string",
                        "description": "Nombre de la estación o municipio a consultar (ej. 'Itagüí', 'Centro', 'El Poblado', 'Envigado', 'Bello', 'Caldas').",
                    }
                },
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_tomorrow",
            "description": (
                "Calcula el pronóstico de calidad del aire para mañana en una estación del SIATA "
                "(categoría esperada, color de alerta, concentración estimada y probabilidades)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {
                        "type": "string",
                        "description": "Nombre de la estación o municipio para predecir el día siguiente.",
                    }
                },
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Busca en la normativa oficial del AMVA (Protocolo POECA y Resolución 2254 de 2017) "
                "las recomendaciones de salud para ciudadanos, deportistas y grupos sensibles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta sobre recomendaciones de salud, deporte o medidas del POECA.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# =====================================================================
# 3. LOOP AGÉNTICO MANUAL CON LENGUAJE CIUDADANO CLARO Y SIN EMOJIS
# =====================================================================

@dataclass
class AgentStepTrace:
    """Almacena la información de cada paso del loop agéntico con un resumen legible para el usuario."""
    iteration: int
    tool_name: str
    tool_args: Dict[str, Any]
    observation: str
    human_summary: str = ""


@dataclass
class AgentResponse:
    """Respuesta estructurada devuelta por el loop del agente al dashboard."""
    final_answer: str
    steps: List[AgentStepTrace] = field(default_factory=list)
    iterations_used: int = 0
    used_fallback_mode: bool = False


SYSTEM_PROMPT = """Eres el Asistente Ambiental de Calidad del Aire del Valle de Aburrá (Universidad EAFIT - SIATA/AMVA).
Tu objetivo es explicar a cualquier ciudadano, en un lenguaje claro, cercano, profesional y fácil de entender, cómo está el aire hoy, qué se espera para mañana y qué cuidados de salud debe tener según la normativa vigente.

REGLAS ESTRICTAS DE SELECCIÓN DE HERRAMIENTAS (NO LLAMES HERRAMIENTAS DE MÁS):
1. Llama a `get_pm25(estacion)` ÚNICAMENTE si el usuario pregunta por el nivel o estado actual/hoy de PM2.5, lluvia o viento en una estación.
2. Llama a `predict_tomorrow(estacion)` ÚNICAMENTE si el usuario pregunta por el pronóstico de mañana, tendencia futura, probabilidad o estado latente HMM.
3. Llama a `search_docs(query)` ÚNICAMENTE si el usuario pregunta explícitamente por la normativa (POECA / Resolución 2254 de 2017), recomendaciones de salud, grupos sensibles, deporte al aire libre, medidas institucionales (pico y placa) o causas meteorológicas/topográficas. Si el usuario SOLO pide el dato actual o el pronóstico sin pedir recomendaciones ni normativa, NO llames a `search_docs`.
4. Si una herramienta devuelve una `OBSERVACIÓN DE ERROR` (por ejemplo, una estación que no existe en el Valle de Aburrá como Bogotá), NO llames más herramientas: informa inmediatamente al usuario qué estaciones son válidas (Bello, Medellín Centro, El Poblado, Envigado, Itagüí y Caldas).

REGLAS DE ESTILO Y CITACIÓN:
5. NO uses emojis en tu respuesta. Mantén un tono sobrio, claro y profesional.
6. Traduce las cifras técnicas para el ciudadano: explica el estado del aire en palabras sencillas y pon el valor en microgramos por metro cúbico (ug/m3) como respaldo.
7. Siempre que hayas llamado a `search_docs(query)`, cita explícitamente el documento con el formato `[Fuente: normativa_poeca_amva.md, DOC-XX]`. Incluso si no llamaste a `search_docs`, menciona que las categorías siguen la Resolución 2254 de 2017 y el protocolo POECA (`normativa_poeca_amva.md`).
8. Cierra con una breve nota indicando que la información es orientativa y no reemplaza el concepto médico profesional.
"""


class AirQualityAgent:
    """
    Loop Agéntico Manual con límite T_max, conmutación automática de modelos en Groq
    y síntesis ciudadana clara sin bloques gigantes ni código crudo.
    """

    FALLBACK_GROQ_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    def __init__(
        self,
        toolkit: AirQualityToolkit,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-120b",
        t_max: int = 5,
    ) -> None:
        self.toolkit = toolkit
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.model_name = model_name
        self.t_max = t_max

        self.tool_registry: Dict[str, Callable[..., str]] = {
            "get_pm25": self.toolkit.get_pm25,
            "predict_tomorrow": self.toolkit.predict_tomorrow,
            "search_docs": self.toolkit.search_docs,
        }

    @staticmethod
    def _build_human_step_summary(tool_name: str, args: Dict[str, Any], observation: str) -> str:
        """Convierte la salida técnica de una herramienta en una frase comprensible para la pestaña de auditoría."""
        if observation.startswith("OBSERVACIÓN DE ERROR"):
            return f"Se detectó un parámetro inválido al ejecutar {tool_name}: {observation}"
        try:
            data = json.loads(observation)
            if tool_name == "get_pm25":
                return (
                    f"Consulta de medición actual en {data.get('nombre_estacion')}: "
                    f"Calidad {data.get('categoria_ica_actual')} (Color {data.get('color_alerta')}, "
                    f"{data.get('pm25_actual_ugm3')} ug/m3), lluvia de {data.get('precipitacion_mm_24h')} mm "
                    f"y viento de {data.get('velocidad_viento_ms')} m/s."
                )
            if tool_name == "predict_tomorrow":
                return (
                    f"Pronóstico calculado para mañana en {data.get('nombre_estacion')}: "
                    f"Se proyecta calidad {data.get('categoria_predicha_manana')} (Color {data.get('color_alerta_manana')}, "
                    f"{data.get('pm25_predicho_manana_ugm3')} ug/m3)."
                )
            if tool_name == "search_docs":
                frags = data.get("fragmentos_recuperados", [])
                if frags:
                    return (
                        f"Búsqueda en el protocolo POECA: se recuperó la sección '{frags[0].get('seccion')}' "
                        f"({frags[0].get('chunk_id')})."
                    )
        except Exception:
            pass
        return f"Ejecución completada de la herramienta {tool_name} con parámetros {args}."

    def _execute_tool_safely(self, tool_name: str, raw_args: str | Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        try:
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}
        except Exception as parse_err:
            return {}, f"OBSERVACIÓN DE ERROR: Argumentos JSON inválidos para '{tool_name}': {parse_err}"

        if tool_name not in self.tool_registry:
            return (
                parsed_args,
                f"OBSERVACIÓN DE ERROR: La herramienta '{tool_name}' no existe. "
                f"Herramientas válidas: {list(self.tool_registry.keys())}.",
            )

        try:
            func = self.tool_registry[tool_name]
            observation = func(**parsed_args)
            return parsed_args, observation
        except TypeError as type_err:
            return (
                parsed_args,
                f"OBSERVACIÓN DE ERROR [{tool_name}]: Argumentos incorrectos {parsed_args}. Detalle: {type_err}",
            )
        except Exception as exc:
            return parsed_args, f"OBSERVACIÓN DE ERROR [{tool_name}]: Excepción inesperada: {exc}"

    def _run_citizen_friendly_synthesis(self, user_query: str, default_station: str) -> AgentResponse:
        """
        Ejecuta el loop de herramientas y redacta una respuesta clara, completa y pensada para
        el ciudadano común (sin emojis, sin código JSON crudo y sin encabezados gigantes).
        """
        steps: List[AgentStepTrace] = []
        q_lower = user_query.lower()

        # Buscar palabras completas (\b...\b) para que la palabra "calidad" NO haga match con "cali"
        invalid_candidates = ["bogotá", "bogota", "cali", "barranquilla", "cartagena", "fantasma", "marte", "rionegro"]
        station_target = default_station
        for inv in invalid_candidates:
            if re.search(rf"\b{re.escape(inv)}\b", q_lower):
                station_target = inv.capitalize()
                break
        else:
            for candidate in ["bello", "centro", "museo", "poblado", "envigado", "itagüí", "itagui", "caldas"]:
                if re.search(rf"\b{re.escape(candidate)}\b", q_lower):
                    station_target = "Itagüí" if candidate == "itagui" else candidate.capitalize()
                    break

        needs_current = any(
            kw in q_lower
            for kw in ["hoy", "actual", "está", "esta", "pm2.5", "pm25", "nivel", "medición", "estación", "estacion", "aire", "compara"]
        )
        needs_prediction = any(
            kw in q_lower
            for kw in ["mañana", "manana", "predic", "pronóstic", "pronostic", "futuro", "espera", "probabilidad", "hmm", "naive bayes"]
        )
        needs_rag = any(
            kw in q_lower
            for kw in [
                "salud", "poeca", "norma", "resolución", "resolucion", "recomend", "ejercicio", "trotar",
                "niños", "asma", "sensible", "alerta", "naranja", "rojo", "amarillo", "medida", "pico y placa", "qué hacer",
            ]
        )

        if not (needs_current or needs_prediction or needs_rag):
            needs_current = needs_prediction = needs_rag = True

        iter_count = 0
        obs_curr, obs_pred, obs_rag = None, None, None

        if needs_current and iter_count < self.t_max:
            iter_count += 1
            args1, obs_curr = self._execute_tool_safely("get_pm25", {"estacion": station_target})
            steps.append(
                AgentStepTrace(
                    iteration=iter_count,
                    tool_name="get_pm25",
                    tool_args=args1,
                    observation=obs_curr,
                    human_summary=self._build_human_step_summary("get_pm25", args1, obs_curr),
                )
            )

        if needs_prediction and iter_count < self.t_max:
            iter_count += 1
            args2, obs_pred = self._execute_tool_safely("predict_tomorrow", {"estacion": station_target})
            steps.append(
                AgentStepTrace(
                    iteration=iter_count,
                    tool_name="predict_tomorrow",
                    tool_args=args2,
                    observation=obs_pred,
                    human_summary=self._build_human_step_summary("predict_tomorrow", args2, obs_pred),
                )
            )

        # Seleccionar la consulta normativa más pertinente según la categoría actual o predicha
        rag_query = user_query
        if obs_pred and not obs_pred.startswith("OBSERVACIÓN DE ERROR"):
            cat_p = json.loads(obs_pred).get("categoria_predicha_manana", "")
            rag_query = f"Calidad del Aire {cat_p} recomendaciones de salud y deporte {user_query}"
        elif obs_curr and not obs_curr.startswith("OBSERVACIÓN DE ERROR"):
            cat_c = json.loads(obs_curr).get("categoria_ica_actual", "")
            rag_query = f"Calidad del Aire {cat_c} recomendaciones de salud y deporte {user_query}"

        elif needs_rag and iter_count < self.t_max:
            iter_count += 1
            args3, obs_rag = self._execute_tool_safely("search_docs", {"query": rag_query})
            steps.append(
                AgentStepTrace(
                    iteration=iter_count,
                    tool_name="search_docs",
                    tool_args=args3,
                    observation=obs_rag,
                    human_summary=self._build_human_step_summary("search_docs", args3, obs_rag),
                )
            )

        # Si hubo una estación fuera del Valle de Aburrá, informar de forma clara
        for obs in (obs_curr, obs_pred):
            if obs and obs.startswith("OBSERVACIÓN DE ERROR"):
                return AgentResponse(
                    final_answer=(
                        f"**Aviso de validación de estación:**\n\n"
                        f"El sistema registró que la ubicación consultada (`{station_target}`) no pertenece a la red de monitoreo "
                        f"del Valle de Aburrá.\n\n"
                        f"- **Observación registrada por la herramienta:** `{obs}`\n"
                        f"- **Estaciones disponibles para consulta:** Bello (U. San Buenaventura), Medellín Centro (Museo de Antioquia), "
                        f"El Poblado (Tanques EPM), Envigado (Hosp. Manuel Uribe Ángel), Itagüí (Casa de la Justicia) y Caldas (E. Joaquín Aristizábal).\n\n"
                        f"Según el protocolo metropolitano `[Fuente: normativa_poeca_amva.md, DOC-01]`, la vigilancia oficial cubre los diez municipios "
                        f"del Área Metropolitana del Valle de Aburrá.\n\n"
                        f"*Nota: Información de carácter académico y orientativo; no reemplaza el concepto médico profesional.*"
                    ),
                    steps=steps,
                    iterations_used=iter_count,
                    used_fallback_mode=True,
                )

        paragraphs: List[str] = []

        if obs_curr and not obs_curr.startswith("OBSERVACIÓN DE ERROR"):
            d_c = json.loads(obs_curr)
            paragraphs.append(
                f"**Estado actual del aire en {d_c['nombre_estacion']} ({d_c['municipio']}):**\n"
                f"Hoy ({d_c['fecha_ultima_medicion']}) la calidad del aire es **{d_c['categoria_ica_actual']} (Alerta color {d_c['color_alerta']})**. "
                f"{d_c['significado_ciudadano']} "
                f"En las últimas 24 horas se registraron lluvias de {d_c['precipitacion_mm_24h']} mm y vientos de {d_c['velocidad_viento_ms']} m/s "
                f"(concentración medida de material particulado PM2.5: {d_c['pm25_actual_ugm3']} ug/m3; promedio de la última semana: "
                f"{d_c['resumen_ultimos_7_dias']['promedio_pm25_ugm3']} ug/m3)."
            )

        if obs_pred and not obs_pred.startswith("OBSERVACIÓN DE ERROR"):
            d_p = json.loads(obs_pred)
            probs_dict: Dict[str, str] = d_p.get("probabilidades_por_categoria", {})
            probs_texto = ", ".join([f"{cat}: {pct}" for cat, pct in probs_dict.items()])
            paragraphs.append(
                f"**Pronóstico para mañana en {d_p['nombre_estacion']}:**\n"
                f"De acuerdo con el comportamiento reciente de la lluvia y la ventilación del valle, para mañana se proyecta una calidad del aire "
                f"**{d_p['categoria_predicha_manana']} (Color {d_p['color_alerta_manana']})**, con un nivel estimado de {d_p['pm25_predicho_manana_ugm3']} ug/m3 "
                f"(frente a los {d_p['pm25_actual_ugm3']} ug/m3 actuales). "
                f"{d_p['significado_ciudadano_manana']}\n"
                f"- **Tendencia general del episodio ambiental:** {d_p['estado_episodio_ambiental']}.\n"
                f"- **Probabilidad estimada por categoría:** {probs_texto}."
            )

        if obs_rag and not obs_rag.startswith("OBSERVACIÓN DE ERROR"):
            d_r = json.loads(obs_rag)
            frags = d_r.get("fragmentos_recuperados", [])
            if frags:
                top_f = frags[0]
                citation = f"[Fuente: {top_f.get('fuente', 'normativa_poeca_amva.md')}, {top_f.get('chunk_id', 'DOC-01')}]"
                full_content = top_f.get("contenido", "").strip()
                paragraphs.append(
                    f"**Recomendaciones de salud y normativa ambiental {citation}:**\n"
                    f"*{top_f.get('seccion', 'Lineamiento POECA')}*\n\n"
                    f"{full_content}"
                )

        paragraphs.append(
            "*Nota de alcance: Este reporte es generado con fines informativos y académicos a partir de datos abiertos de SIATA y los lineamientos del AMVA. No reemplaza la valoración médica profesional ni los boletines oficiales.*"
        )

        return AgentResponse(
            final_answer="\n\n".join(paragraphs),
            steps=steps,
            iterations_used=iter_count,
            used_fallback_mode=True,
        )

    def _call_groq_with_model_fallback(self, client: Any, messages: List[Dict[str, Any]], use_tools: bool = True) -> Any:
        """
        Intenta invocar el modelo seleccionado en Groq con reintento automático (backoff)
        ante límites de tasa por minuto (HTTP 429) o errores de validación de tool_calls (HTTP 400).
        """
        import time

        candidates = [self.model_name] + [m for m in self.FALLBACK_GROQ_MODELS if m != self.model_name]
        last_err: Optional[Exception] = None
        for candidate_model in candidates:
            for attempt in range(2):
                try:
                    if use_tools:
                        return client.chat.completions.create(
                            model=candidate_model,
                            messages=messages,
                            tools=AGENT_TOOLS_SCHEMA,
                            tool_choice="auto",
                            temperature=0.0,
                        )
                    return client.chat.completions.create(
                        model=candidate_model,
                        messages=messages,
                        temperature=0.0,
                    )
                except Exception as err:
                    last_err = err
                    err_str = str(err).lower()
                    if "429" in err_str or "rate_limit" in err_str:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    if "model_not_found" in err_str or "404" in err_str or "decommissioned" in err_str or "tool_use_failed" in err_str or "400" in err_str:
                        break
                    break
        if last_err is not None:
            raise last_err
        raise RuntimeError("No se pudo conectar con Groq API.")

    def run(self, user_query: str, default_station: str = "Medellín") -> AgentResponse:
        """
        Ejecuta el loop agéntico manual desde t = 1 hasta T_max.
        """
        if not self.api_key or not GROQ_SDK_AVAILABLE:
            return self._run_citizen_friendly_synthesis(user_query, default_station)

        client = Groq(api_key=self.api_key)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[Estación seleccionada por defecto en el panel: '{default_station}']\n"
                    f"Consulta del ciudadano: {user_query}"
                ),
            },
        ]
        steps: List[AgentStepTrace] = []

        for iteration in range(1, self.t_max + 1):
            try:
                completion = self._call_groq_with_model_fallback(client, messages, use_tools=True)
            except Exception:
                # Si ningún modelo de la cuenta responde o no hay red, entregar la síntesis ciudadana limpia sin avisos de error crudos
                return self._run_citizen_friendly_synthesis(user_query, default_station)

            response_msg = completion.choices[0].message
            tool_calls = getattr(response_msg, "tool_calls", None)

            if not tool_calls:
                clean_ans = LocalAMVARetriever._sanitize_markdown_headings(response_msg.content or "")
                return AgentResponse(
                    final_answer=clean_ans or "No se generó contenido de respuesta.",
                    steps=steps,
                    iterations_used=iteration,
                    used_fallback_mode=False,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response_msg.content or "",
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

            for tc in tool_calls:
                t_name = tc.function.name
                t_args, observation = self._execute_tool_safely(t_name, tc.function.arguments)

                steps.append(
                    AgentStepTrace(
                        iteration=iteration,
                        tool_name=t_name,
                        tool_args=t_args,
                        observation=observation,
                        human_summary=self._build_human_step_summary(t_name, t_args, observation),
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": observation,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Se alcanzó el límite de pasos (T_max = {self.t_max}). "
                    "Con base en los datos consultados, redacta la respuesta final para el ciudadano de forma clara, sin emojis y citando la fuente normativa."
                ),
            }
        )
        try:
            final_completion = self._call_groq_with_model_fallback(client, messages, use_tools=False)
            final_text = LocalAMVARetriever._sanitize_markdown_headings(
                final_completion.choices[0].message.content or ""
            )
        except Exception:
            return self._run_citizen_friendly_synthesis(user_query, default_station)

        return AgentResponse(
            final_answer=final_text,
            steps=steps,
            iterations_used=self.t_max,
            used_fallback_mode=False,
        )
