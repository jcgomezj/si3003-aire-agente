# 🌫️ Agente de calidad del aire del Valle de Aburrá

Proyecto final integrador — **SI3003 Inteligencia Artificial** (EAFIT, 2026-1).

Solución de IA sobre datos reales colombianos que combina:

- **Parte 1 (fundamentos clásicos):** modelo propio para predecir la categoría
  de PM2.5 del día siguiente, con métrica frente a una línea base.
- **Parte 2 (IA moderna):** agente basado en LLM con **RAG** y **3 herramientas**
  (tool calling), siguiendo un loop agéntico manual.
- **Dashboard interactivo** (Streamlit) con mapa de estaciones, serie de tiempo
  y chat con el agente.

> El problema: los ciudadanos no comprenden qué significa el PM2.5 del día ni
> qué hacer al respecto. Este agente lo explica y recomienda acciones citando
> normativa, protocolos de episodios críticos y guías de salud.

---

## 🏗️ Arquitectura

```
si3003-aire-agente/
├── app.py                 # Dashboard Streamlit (UI + chat)
├── requirements.txt
├── .env.example           # plantilla de variables de entorno
├── src/
│   ├── config.py          # configuración central (LLM, rutas, umbrales PM2.5)
│   ├── agent/
│   │   ├── llm_client.py  # cliente OpenAI-compatible (Groq/NIM/Ollama)
│   │   └── agent.py       # LOOP AGÉNTICO manual (ReAct + tool calling)
│   ├── rag/
│   │   ├── indexer.py     # trocea el corpus -> ChromaDB (embeddings locales)
│   │   └── retriever.py   # recupera fragmentos citables
│   ├── tools/
│   │   └── tools.py       # get_pm25 · predict_tomorrow · search_docs
│   ├── data/
│   │   └── loader.py      # carga SIATA (CSV real o sintético de respaldo)
│   └── classic/
│       └── predictor.py   # MODELO CLÁSICO de la Parte 1 (interfaz)
├── corpus/                # documentos para el RAG (normativa, salud, protocolos)
├── data/                  # datos crudos/procesados + índice Chroma (ignorado en git)
├── eval/                  # preguntas de prueba fijas + script de evaluación
└── docs/                  # declaración de uso de IA, informe
```

---

## 🚀 Puesta en marcha (reproducible, sin intervención manual)

### 1. Requisitos
- Python 3.11+
- Una API key de un proveedor LLM gratuito. Recomendado: **Groq**
  (https://console.groq.com/keys).

### 2. Instalación
```bash
git clone https://github.com/jcgomezj/si3003-aire-agente.git
cd si3003-aire-agente
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar el LLM
```bash
cp .env.example .env
# edita .env y pon tu LLM_API_KEY
```

### 4. (Opcional) Construir el índice RAG
Se construye solo en el primer arranque, pero puedes forzarlo:
```bash
python -m src.rag.indexer
```

### 5. Ejecutar el dashboard
```bash
streamlit run app.py
```

---

## 🧪 Evaluación del agente
```bash
python -m eval.run_eval
```
Ejecuta las preguntas de prueba fijas y reporta si el agente usó la herramienta
correcta y citó la fuente esperada. Ver `eval/preguntas_evaluacion.md`.

---

## 📊 Datos

- **Fuente:** SIATA / Área Metropolitana del Valle de Aburrá (estaciones de
  calidad del aire y meteorología).
- **Plan B:** si la API no está disponible, se usan históricos descargados en
  `data/raw/siata_pm25.csv`. Si no existe ningún CSV, el sistema genera un
  dataset sintético reproducible para que el pipeline corra de punta a punta
  (⚠️ reemplazar por datos reales antes de la entrega — responsable: datos).

---

## 🔧 Herramientas del agente (tool calling)

| Herramienta | Función |
|-------------|---------|
| `get_pm25(estacion)` | Último PM2.5 y categoría de una estación. |
| `predict_tomorrow(estacion)` | Predicción del día siguiente (modelo clásico Parte 1). |
| `search_docs(query)` | Recupera del corpus RAG para fundamentar y **citar** fuentes. |

El loop tiene un tope de pasos (`AGENT_MAX_STEPS`, por defecto 6) y devuelve los
errores de las herramientas como observación, según las recomendaciones del
enunciado.

---

## 🤖 Proveedores LLM soportados
Cualquier endpoint compatible con OpenAI. Cambia solo `.env`:
- **Groq** (por defecto) — `llama-3.3-70b-versatile`
- **NVIDIA NIM**
- **Ollama** (local)

---

## 📄 Declaración de uso de IA
Ver [`docs/DECLARACION_IA.md`](docs/DECLARACION_IA.md).

## 👥 Equipo
- **Juan Camilo Gómez** (jcgomezj) — Agente LLM, RAG, dashboard e integración.
- **_(compañero)_** — Datos, modelo clásico (Parte 1), EDA y visualizaciones.
