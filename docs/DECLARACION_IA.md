# Declaración de uso de IA generativa

Conforme al pacto pedagógico de SI3003, declaramos el uso de herramientas de
inteligencia artificial generativa en este proyecto.

## Herramientas utilizadas y propósito

| Herramienta | Propósito |
|-------------|-----------|
| _(p. ej. ChatGPT / Claude / Kiro)_ | Andamiaje del código (estructura del proyecto, loop del agente, dashboard), redacción de documentación y depuración. |
| Modelo LLM del agente (Groq / NVIDIA NIM / Ollama) | **Componente funcional del producto**: es el motor del agente conversacional; no se usó para generar el informe. |

## Alcance y comprensión

- El equipo comprende y puede explicar cada componente entregado: el modelo
  clásico de la Parte 1, el loop agéntico, el RAG, las herramientas y el
  dashboard.
- La IA generativa se usó como asistente de desarrollo; las decisiones de
  diseño, la validación de resultados y la integración fueron realizadas y
  verificadas por el equipo.
- No hubo dependencia total de la herramienta: todo el código fue revisado,
  probado y adaptado al problema específico.

## Datos

- Se usaron datos reales colombianos (SIATA / AMVA) con Plan B de históricos
  en CSV. Ver `README.md`, sección de datos.
- No se procesaron datos personales ni individuales.

---

_Integrantes:_
- Juan Camilo Gómez (jcgomezj) — Agente LLM, RAG, dashboard e integración.
- _(Nombre del compañero)_ — Datos, modelo clásico, EDA y visualizaciones.
