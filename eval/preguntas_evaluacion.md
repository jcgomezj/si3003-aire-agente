# Evaluación del agente — preguntas de prueba fijas

Conjunto de preguntas para evaluar si el agente (1) llama a la herramienta
correcta y (2) cita la fuente del corpus cuando corresponde. Marcar ✅/❌ tras
cada ejecución. Ver `eval/run_eval.py` para ejecutarlas automáticamente.

| # | Pregunta | Herramienta(s) esperada(s) | ¿Debe citar fuente? |
|---|----------|----------------------------|---------------------|
| 1 | ¿Cómo está el aire hoy en Medellín-Centro? | get_pm25 | No |
| 2 | ¿Qué categoría de PM2.5 habrá mañana en Itagüí? | predict_tomorrow | No |
| 3 | ¿Qué significa que el PM2.5 sea "dañino a grupos sensibles"? | search_docs | Sí (normativa_pm25.md) |
| 4 | ¿Qué debo hacer si el aire está dañino para la salud? | search_docs | Sí (recomendaciones_salud.md) |
| 5 | ¿Qué es un estado de alerta en el protocolo de episodios? | search_docs | Sí (protocolo_episodios.md) |
| 6 | Dime el PM2.5 de Bello y qué recomendaciones aplican. | get_pm25 + search_docs | Sí |
| 7 | ¿Puedo salir a trotar mañana en Envigado? | predict_tomorrow + search_docs | Sí |
| 8 | ¿Cuál es el PM2.5 en la estación "Marte"? (no existe) | get_pm25 (manejo de error) | No |
| 9 | ¿Qué medidas se toman durante una emergencia por mala calidad del aire? | search_docs | Sí (protocolo_episodios.md) |
| 10 | ¿Quiénes son los grupos sensibles a la contaminación? | search_docs | Sí (recomendaciones_salud.md) |
| 11 | Compara el aire de hoy en Caldas con el umbral de salud. | get_pm25 + search_docs | Sí |
| 12 | ¿Debería usar tapabocas hoy en Medellín-Centro? | get_pm25 + search_docs | Sí |

> Objetivo mínimo: el agente acierta la herramienta en >= 80% de los casos y
> cita la fuente en todas las preguntas normativas/de salud.
