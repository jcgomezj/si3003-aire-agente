"""Carga de datos de calidad del aire (SIATA) con Plan B.

Estrategia:
  1. Si existe un CSV en data/raw/ (histórico descargado -> Plan B), se usa.
  2. Si no existe, se genera un dataset SINTÉTICO reproducible para que el
     esqueleto corra de punta a punta. Debe reemplazarse por datos reales
     (lo hace tu compañero, responsable de datos).

El CSV esperado tiene columnas:
    fecha, estacion, lat, lon, pm25, temperatura, humedad, viento, lluvia
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

CSV_NAME = "siata_pm25.csv"

# Estaciones de ejemplo del Valle de Aburrá (coordenadas aproximadas).
# Reemplazar por el catálogo real de SIATA.
ESTACIONES = {
    "Medellín-Centro": (6.2518, -75.5636),
    "Itagüí": (6.1716, -75.6116),
    "Envigado": (6.1690, -75.5820),
    "Bello": (6.3378, -75.5600),
    "Caldas": (6.0914, -75.6366),
}


def _generate_synthetic(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Genera datos sintéticos reproducibles (Plan de respaldo del esqueleto)."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    fechas = pd.date_range(end - pd.Timedelta(days=n_days - 1), end, freq="D")

    rows = []
    for est, (lat, lon) in ESTACIONES.items():
        base = rng.uniform(15, 30)  # nivel base por estación
        for i, f in enumerate(fechas):
            # estacionalidad + ruido + episodios ocasionales
            season = 8 * np.sin(2 * np.pi * i / 30)
            spike = rng.choice([0, 0, 0, 0, 25], p=[0.9, 0.0, 0.0, 0.0, 0.1])
            pm25 = max(2.0, base + season + rng.normal(0, 5) + spike)
            rows.append(
                {
                    "fecha": f,
                    "estacion": est,
                    "lat": lat,
                    "lon": lon,
                    "pm25": round(pm25, 1),
                    "temperatura": round(rng.uniform(18, 28), 1),
                    "humedad": round(rng.uniform(50, 90), 1),
                    "viento": round(rng.uniform(0.5, 4.0), 1),
                    "lluvia": round(max(0, rng.normal(3, 5)), 1),
                }
            )
    return pd.DataFrame(rows)


def load_air_quality() -> pd.DataFrame:
    """Carga el dataset de calidad del aire (CSV real o sintético de respaldo)."""
    csv_path = config.DATA_RAW / CSV_NAME
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["fecha"])
    else:
        df = _generate_synthetic()
        # Persistimos el sintético para que el pipeline sea reproducible
        config.DATA_RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    df["categoria"] = df["pm25"].apply(config.pm25_category)
    return df


def list_estaciones() -> list[str]:
    return list(ESTACIONES.keys())


if __name__ == "__main__":
    data = load_air_quality()
    print(data.head())
    print(f"\nFilas: {len(data)} | Estaciones: {data['estacion'].nunique()}")
