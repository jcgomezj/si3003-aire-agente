# Sistema Inteligente de Calidad del Aire del Valle de Aburrá (SI3003 - EAFIT)

**Universidad EAFIT — Escuela de Ciencias Aplicadas e Ingeniería**  
**Asignatura:** SI3003 – Inteligencia Artificial (Periodo 2026-1)  
**Profesor:** Jorge Iván Padilla Buriticá  
**Autores:** Juan Camilo Gómez Jiménez y Pablo Arango  
**Repositorio Oficial:** https://github.com/jcgomezj/si3003-aire-agente  

Proyecto final integrador que desarrolla la opción **3.1 Agente de calidad del aire del Valle de Aburrá**, combinando técnicas clásicas de IA (Parte 1) con un agente conversacional basado en LLM, RAG y *tool calling* (Parte 2) en un dashboard interactivo en Streamlit.

---

## Arquitectura del Proyecto

```text
Proyecto_Final_IA/
├── data/
│   └── siata_pm25_historico.csv       # Datos históricos verificados del SIATA (Plan B resiliente)
├── docs_rag/
│   └── normativa_poeca_amva.md        # Corpus normativo: Acuerdo POECA del AMVA y Res. 2254 de 2017
├── entregables_pdf/
│   ├── Informe_breve_calidad_aire_SI3003_estilo_original.pdf # Entregable 2: Informe técnico oficial (4 páginas)
│   ├── 2_Evaluacion_Agente_12_Preguntas_SI3003.pdf           # Entregable 4: Evaluación de las 12 preguntas del agente
│   └── 3_Declaracion_Uso_IA_Generativa_SI3003.pdf            # Entregable 5: Declaración de uso ético de IA generativa
├── models/
│   └── air_quality_model_part1.pkl    # Artefacto serializado del modelo clásico de la Parte 1 (se genera automáticamente)
├── notebooks/
│   └── 01_eda_y_modelo_parte1.ipynb   # Notebook interactivo con EDA, métricas Parte 1 y evaluación del Agente
├── src/
│   ├── data_loader.py                 # Ingeniería de datos: conexión API / Plan B CSV, limpieza e imputación
│   ├── eda.py                         # Análisis Exploratorio de Datos y generación de visualizaciones Plotly
│   ├── model_part1.py                 # Parte 1: Gaussian Naive Bayes + Regresión Ridge + Filtro HMM en NumPy
│   ├── agent_rag.py                   # Parte 2: RAG con FAISS + Loop Agéntico Manual (T_max) + 3 Herramientas
│   ├── evaluate_agent.py              # Entregable 4: Suite de evaluación automática con 12 preguntas de prueba
│   └── app.py                         # Dashboard interactivo en Streamlit (Mapa, Serie de Tiempo, Chat y Evaluación)
├── INFORME_PROYECTO.md                # Entregable 2: Informe técnico estructurado (máximo 4 páginas)
├── run_all.py                         # Orquestador de un solo comando para ejecutar todo el proyecto
├── requirements.txt                   # Dependencias reproducibles
└── README.md                          # Instrucciones de instalación, ejecución y declaración de IA
```

### Integración No Trivial Parte 1 + Parte 2 y Transparencia de Datos
1. **Transparencia sobre los Datos (`src/data_loader.py` y `data/siata_pm25_historico.csv`):**
   - Por defecto (`prefer_local_plan_b=True`), el pipeline utiliza `data/siata_pm25_historico.csv` ($360$ registros diarios = $60$ días $\times\ 6$ estaciones del Valle de Aburrá), el cual es un **dataset sintético con acoplamiento físico** generado de forma determinista (`seed=42`) en `generate_enriched_historical_dataset()` para servir como **Plan B reproducible** ante fallas de conectividad o cambios en APIs externas. Modela las coordenadas reales de las estaciones del SIATA (*Bello, Medellín Centro, El Poblado, Envigado, Itagüí y Caldas*), la persistencia temporal del $PM_{2.5}$, el lavado atmosférico por lluvia y la ventilación por viento.
   - **Conexión con la fuente real del SIATA / `datos.gov.co`:** La función `try_fetch_siata_or_datos_gov()` implementa la consulta al recurso Socrata en `datos.gov.co` y se activa pasando `load_air_quality_data(prefer_local_plan_b=False)`, o bien reemplazando `data/siata_pm25_historico.csv` con históricos descargados del portal oficial de datos abiertos del SIATA (`https://siata.gov.co/descarga_siata/`).
2. **Parte 1 (`src/model_part1.py` — Métricas Reproducibles en Test Temporal):**
   - Implementación propia desde cero en NumPy de un clasificador **Gaussian Naive Bayes**, una **Regresión Ridge ($L_2$) en forma cerrada** y un **Filtro de Markov (HMM)** de 3 estados latentes (*Normal*, *Alerta / Prevención*, *Episodio Crítico*).
   - Sobre los **$N_{\text{total}} = 354$ pares supervisados de transición ($t \rightarrow t+1$)** divididos cronológicamente en **$N_{\text{train}} = 265$ (75%)** y **$N_{\text{test}} = 89$ (25%)**, el modelo obtiene:
     - **Accuracy en Test ($N_{\text{test}} = 89$):** `78.65%` (superando al Baseline de Persistencia `71.91%` y a la Clase Mayoritaria `43.82%`).
     - **Macro F1-Score:** `0.7392` (vs. `0.7169` en Persistencia y `0.2031` en Mayoritaria).
     - **Weighted F1-Score:** `0.7873` (vs. `0.7195` en Persistencia).
     - **MAE / RMSE Regresión $PM_{2.5}$:** `MAE = 3.26 µg/m³` (vs. `4.25 µg/m³` en Persistencia) y `RMSE = 4.03 µg/m³`.
3. **Parte 2 (`src/agent_rag.py` y `src/evaluate_agent.py`):**
   - **Loop Agéntico Manual:** Controla explícitamente el ciclo $\text{Pensamiento} \rightarrow \text{Llamada a Herramienta} \rightarrow \text{Observación}$ con límite $T_{\max}$ usando la API de **Groq** (`openai/gpt-oss-120b` / `llama-3.3-70b-versatile`).
   - **Tres herramientas con manejo robusto de errores:** `get_pm25(estacion)`, `predict_tomorrow(estacion)` y `search_docs(query)` (con **FAISS** `IndexFlatIP`).
   - **Evaluación real con criterio estricto de igualdad de conjuntos (`src/evaluate_agent.py`):** Evaluando el **Agente LLM Real en Groq (`openai/gpt-oss-120b`)** sobre las 12 preguntas de prueba (`data/evaluacion_agente_12_preguntas.csv`), alcanza **91.7% (11/12)** de coincidencia exacta de conjuntos de herramientas (`set(invocadas) == set(esperadas)`): acierta exactamente en `Q01`–`Q10` y `Q12` (incluyendo autorrecuperación ante observación de error en `Q04`), mientras que en `Q11` (*Bogotá Norte*) rechaza directamente la estación fuera del Valle sin llamar a `get_pm25` (`Faltantes: get_pm25`).

---

## Instalación y Ejecución Reproducible

1. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   python -m venv .venv
   # En Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # En Linux / macOS:
   source .venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar la API Key de Groq (opcional gracias al archivo `.env` y al modo Plan B integrado):**
   ```powershell
   # En Windows (PowerShell):
   $env:GROQ_API_KEY="gsk_tu_api_key_de_groq"
   ```
   *(También puedes definirla en un archivo `.env` en la raíz del proyecto o ingresarla directamente desde la barra lateral del dashboard en Streamlit. Si no se proporciona clave, el agente activa su loop determinista local para que la demostración nunca falle).*

4. **Ejecutar TODO el proyecto completo con un solo comando (Datos + EDA + Modelo Parte 1 + Evaluación + Dashboard):**
   ```powershell
   python run_all.py
   ```
   O bien iniciar directamente el Dashboard interactivo en Streamlit:
   ```powershell
   streamlit run src/app.py
   ```

---

## Limitaciones y Declaración de Uso de IA Generativa

- **Descargo de responsabilidad:** Este sistema es un prototipo académico para el curso SI3003 y **no constituye consejo médico, legal ni un canal oficial de alerta temprana**.
- **Declaración de uso de IA generativa:** Se utilizó asistencia de IA generativa para estructurar la plantilla modular del proyecto, las funciones de visualización en Plotly/Streamlit y la documentación base, verificando manualmente la formulación matemática del modelo de la Parte 1 y el manejo de errores del loop agéntico conforme al pacto pedagógico.

