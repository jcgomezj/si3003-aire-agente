"""Interfaz del MODELO CLÁSICO (Parte 1).

IMPORTANTE: este módulo define el CONTRATO que el agente consume vía la
herramienta `predict_tomorrow`. La implementación "de verdad" (HMM, red
bayesiana o clasificador propio con métrica vs. línea base) la desarrolla
el compañero responsable de la Parte 1.

Mientras tanto, aquí hay una línea base simple (persistencia/promedio móvil)
para que el pipeline corra de punta a punta. Cuando el compañero entregue su
modelo, basta con reemplazar `predict_category` manteniendo la misma firma.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data.loader import load_air_quality


def predict_category(estacion: str, df: pd.DataFrame | None = None) -> dict:
    """Predice la categoría de PM2.5 del día siguiente para una estación.

    Línea base (placeholder): promedio móvil de los últimos 3 días.
    Devuelve un dict con la predicción y metadatos para que el agente
    pueda explicar y citar el método.
    """
    if df is None:
        df = load_air_quality()

    sub = df[df["estacion"] == estacion].sort_values("fecha")
    if sub.empty:
        return {"error": f"No hay datos para la estación '{estacion}'."}

    ventana = sub["pm25"].tail(3)
    pred_pm25 = round(float(ventana.mean()), 1)
    return {
        "estacion": estacion,
        "pm25_predicho": pred_pm25,
        "categoria_predicha": config.pm25_category(pred_pm25),
        "metodo": "linea_base_promedio_movil_3d",
        "nota": "Placeholder. Se reemplaza por el modelo clásico de la Parte 1.",
    }


if __name__ == "__main__":
    print(predict_category("Medellín-Centro"))
