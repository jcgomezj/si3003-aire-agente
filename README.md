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

### Integración No Trivial Parte 1 + Parte 2
1. **Parte 1 (`src/model_part1.py`):** Implementación propia desde cero en NumPy de un clasificador **Gaussian Naive Bayes** que modela $P(\text{Categoría}_{t+1} \mid PM_{2.5,t}, \text{Lluvia}_t, \text{Viento}_t)$, una regresión regularizada L2 para estimar el valor esperado de $PM_{2.5, t+1}$ y un **Filtro de Markov (HMM)** de 3 estados latentes (*Normal*, *Alerta / Prevención*, *Episodio Crítico*). Se evalúa cuantitativamente con *Accuracy* y *MAE* frente a una línea base de persistencia ($PM_{2.5, t+1} = PM_{2.5, t}$).
2. **Parte 2 (`src/agent_rag.py`):**
   - **Loop Agéntico Manual:** Controla explícitamente el ciclo $\text{Pensamiento} \rightarrow \text{Llamada a Herramienta} \rightarrow \text{Observación}$ con un límite estricto de iteraciones $T_{\max}$ (por defecto $T_{\max}=5$) usando la API de **Groq** (`llama-3.3-70b-versatile`).
   - **Tres herramientas con manejo robusto de errores:**
     1. `get_pm25(estacion)`: Consulta mediciones actuales y resumen reciente de estaciones del SIATA.
     2. `predict_tomorrow(estacion)`: Invoca el modelo clásico de la Parte 1 y devuelve la categoría predicha, probabilidades de clase y estado latente HMM.
     3. `search_docs(query)`: Realiza búsqueda vectorial con **FAISS** (`IndexFlatIP`) sobre `docs_rag/normativa_poeca_amva.md` devolviendo citas verificables.
   - **Manejo de errores como observación:** Cualquier fallo de validación o estación inexistente es capturado y retornado como `OBSERVACIÓN DE ERROR [...]` para que el LLM se recupere sin interrumpir la aplicación.

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

