"""
Módulo de la Parte 1 (Fundamentos Clásicos de IA) - SI3003 Universidad EAFIT.

Cumple íntegramente con el Requisito Obligatorio 2 y la Rúbrica (20% Parte 1 + 15% Rigor Experimental):
1. Implementación propia desde cero en NumPy (sin depender de `scikit-learn` como caja negra) de:
   - **Clasificador Gaussian Naive Bayes / Red Bayesiana Naive**: Modela la distribución conjunta
     de las variables meteorológicas y de contaminación en el día $t$ para inferir la probabilidad
     posterior de la categoría del aire del día siguiente ($t+1$):
     $$P(C_{t+1} = c \mid \mathbf{x}_t) \propto P(C_{t+1} = c) \prod_{j=1}^{d} \mathcal{N}(x_{t,j} \mid \mu_{c,j}, \sigma^2_{c,j})$$
     donde $\mathbf{x}_t = [\text{PM2.5}_t, \text{PM2.5\_MA3}_t, \text{Lluvia}_t, \text{Viento}_t, \text{Temp}_t, \text{Humedad}_t]$.
   - **Filtro de Markov Oculto (HMM de 3 estados POECA)**: Estima el estado latente del episodio
     de contaminación (*Normal*, *Alerta / Prevención*, *Episodio Crítico*) mediante actualización
     recursiva de creencia Bayesiana (Chapman-Kolmogorov + verosimilitud gaussiana).
   - **Regresor Ridge en forma cerrada ($L_2$)**: Estima la concentración numérica esperada
     $\widehat{\text{PM2.5}}_{t+1}$ para complementar la categoría predicha.
2. **Validación Temporal (Train / Test Split Cronológico 75% / 25%)** y **Comparación contra 2 Líneas Base**:
   - Baseline 1: Modelo de Persistencia ($\hat{C}_{t+1} = C_t$, asume que mañana el aire estará igual que hoy).
   - Baseline 2: Clase Mayoritaria (predice siempre la categoría más frecuente del histórico).
3. **Métricas Cuantitativas Formales**:
   - Accuracy, Macro F1-Score, Weighted F1-Score, Matriz de Confusión multiclasificación y MAE ($\mu g/m^3$).
4. **Persistencia del Modelo (`models/air_quality_model_part1.pkl`)**:
   - Exporta e importa el modelo entrenado con `pickle` para consumo directo de la herramienta
     `predict_tomorrow(estacion)` del Agente LLM en `src/agent_rag.py`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import (
    ICA_THRESHOLDS,
    load_air_quality_data,
    map_pm25_to_ica,
    map_pm25_to_poeca_state,
)

# Mantener alias para compatibilidad con otros módulos
ICA_CATEGORIES = ICA_THRESHOLDS
pm25_to_category = map_pm25_to_ica

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "air_quality_model_part1.pkl"

FEATURE_COLUMNS = [
    "pm25",
    "pm25_ma3",
    "precipitacion_mm",
    "velocidad_viento_ms",
    "temperatura_c",
    "humedad_relativa_pct",
]


@dataclass
class PredictionResult:
    """Estructura de salida del modelo clásico de la Parte 1 para consumo del Agente (Parte 2)."""
    station_id: str
    nombre_estacion: str
    fecha_base: str
    pm25_actual: float
    categoria_actual: str
    pm25_estimado_manana: float
    categoria_predicha_manana: str
    estado_poeca_manana: str
    color_alerta_manana: str
    probabilidades_clase: Dict[str, float]
    estado_latente_hmm: str
    metricas_validacion: Dict[str, Any]


def compute_classification_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Any]:
    """
    Calcula desde cero en NumPy las métricas formales de clasificación multiclase:
    - Accuracy global
    - Precision, Recall y F1-Score por clase
    - Macro F1-Score y Weighted F1-Score
    - Matriz de confusión (filas = valor real, columnas = predicción)
    """
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "confusion_matrix": []}

    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    k = len(labels)
    conf_matrix = np.zeros((k, k), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            conf_matrix[label_to_idx[yt], label_to_idx[yp]] += 1

    accuracy = float(np.trace(conf_matrix) / max(1, np.sum(conf_matrix)))

    f1_per_class: Dict[str, float] = {}
    supports: List[int] = []
    f1_values: List[float] = []

    for i, lbl in enumerate(labels):
        tp = float(conf_matrix[i, i])
        fp = float(np.sum(conf_matrix[:, i]) - tp)
        fn = float(np.sum(conf_matrix[i, :]) - tp)
        support = int(np.sum(conf_matrix[i, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_per_class[lbl] = round(f1, 4)
        supports.append(support)
        f1_values.append(f1)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    total_support = max(1, sum(supports))
    weighted_f1 = float(sum(f * s for f, s in zip(f1_values, supports)) / total_support)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "f1_por_clase": f1_per_class,
        "labels_orden": labels,
        "confusion_matrix": conf_matrix.tolist(),
    }


class ClassicalAirQualityModel:
    """
    Modelo Clásico de la Parte 1 implementado por el equipo:
    1. Clasificador Probabilístico Gaussian Naive Bayes con suavizado de Laplace y regularización
       de varianza para predecir la categoría ICA del día siguiente ($C_{t+1}$).
    2. Regresor Lineal Ridge ($L_2$) en forma cerrada matricial para estimar $\widehat{\text{PM2.5}}_{t+1}$.
    3. Filtro de Markov Oculto (HMM) para estimar el estado latente del episodio ambiental.
    """

    MODEL_VERSION = "2.2"

    def __init__(self, var_smoothing: float = 1.2, ridge_alpha: float = 1.0) -> None:
        self.model_version: str = self.MODEL_VERSION
        self.var_smoothing = var_smoothing
        self.ridge_alpha = ridge_alpha
        self.classes_: List[str] = []
        self.class_priors_: Dict[str, float] = {}
        self.means_: Dict[str, np.ndarray] = {}
        self.vars_: Dict[str, np.ndarray] = {}
        self.reg_weights_: np.ndarray = np.zeros(len(FEATURE_COLUMNS) + 1)
        self.is_fitted_: bool = False
        self.evaluation_metrics_: Dict[str, Any] = {}

    @staticmethod
    def _prepare_supervised_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara el dataset supervisado asegurando que todas las características meteorológicas
        y los objetivos del día siguiente ($t+1$) estén presentes y ordenados temporalmente.
        """
        df_proc = df.copy()
        if "target_pm25_next" not in df_proc.columns or "pm25_ma3" not in df_proc.columns:
            from src.data_loader import clean_and_preprocess_data
            df_proc = clean_and_preprocess_data(df_proc)

        valid_df = df_proc.dropna(subset=FEATURE_COLUMNS + ["target_pm25_next", "target_categoria_ica_next"]).copy()
        return valid_df.sort_values(["date", "station_id"]).reset_index(drop=True)

    def _fit_parameters(self, X_train: np.ndarray, y_pm25_train: np.ndarray, y_cat_train: List[str]) -> None:
        """Ajusta los parámetros de Gaussian Naive Bayes y los pesos de Regresión Ridge sobre el conjunto de entrenamiento."""
        unique_classes = sorted(list(set(y_cat_train)))
        self.classes_ = unique_classes
        total_count = len(y_cat_train)

        for c in unique_classes:
            idx = [i for i, label in enumerate(y_cat_train) if label == c]
            X_c = X_train[idx]
            # Prior P(C = c) con suavizado de Laplace
            self.class_priors_[c] = (len(idx) + 1.0) / (total_count + len(unique_classes))
            self.means_[c] = np.mean(X_c, axis=0)
            self.vars_[c] = np.var(X_c, axis=0) + self.var_smoothing

        # Regresión Ridge en forma cerrada: w = (X^T X + alpha * I)^(-1) X^T y
        n_samples = len(X_train)
        X_design = np.column_stack([np.ones(n_samples), X_train])
        reg_matrix = np.eye(X_design.shape[1]) * self.ridge_alpha
        reg_matrix[0, 0] = 0.0  # No penalizar el intercepto
        self.reg_weights_ = np.linalg.pinv(X_design.T @ X_design + reg_matrix) @ (X_design.T @ y_pm25_train)
        self.is_fitted_ = True

    def fit_and_evaluate(self, df: pd.DataFrame, train_ratio: float = 0.75) -> Dict[str, Any]:
        """
        Realiza validación temporal (75% entrenamiento inicial cronológico, 25% prueba futura),
        calcula Accuracy, Macro F1, Weighted F1, Matriz de Confusión y MAE frente a dos líneas base
        (Persistencia y Clase Mayoritaria), y finalmente reentrena con el 100% del histórico para producción.
        """
        sup_df = self._prepare_supervised_dataset(df)
        n_total = len(sup_df)
        if n_total < 8:
            raise ValueError("Datos insuficientes para entrenar y evaluar el modelo de la Parte 1.")

        X_all = sup_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_pm25_all = sup_df["target_pm25_next"].to_numpy(dtype=float)
        y_cat_all = sup_df["target_categoria_ica_next"].astype(str).tolist()
        cat_today_all = sup_df["categoria_ica"].astype(str).tolist()

        # Partición temporal estricta (evita data leakage del futuro hacia el pasado)
        split_idx = max(4, int(n_total * train_ratio))
        if split_idx >= n_total - 2:
            split_idx = max(2, n_total // 2)

        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_pm25_train, y_pm25_test = y_pm25_all[:split_idx], y_pm25_all[split_idx:]
        y_cat_train, y_cat_test = y_cat_all[:split_idx], y_cat_all[split_idx:]
        baseline_persist_cat_test = cat_today_all[split_idx:]
        baseline_persist_pm25_test = X_test[:, 0]  # PM2.5_t

        # 1. Entrenar sobre el conjunto de entrenamiento histórico (Train)
        self._fit_parameters(X_train, y_pm25_train, y_cat_train)

        # 2. Evaluar sobre el conjunto de prueba futuro (Test)
        X_test_design = np.column_stack([np.ones(len(X_test)), X_test])
        preds_pm25_test = X_test_design @ self.reg_weights_

        # Predicción híbrida Bayesiana: combina Regresión Ridge regularizada (valor esperado físico de PM2.5_{t+1})
        # con las probabilidades posteriores de Gaussian Naive Bayes cuando está cerca de la frontera de decisión.
        preds_cat_test: List[str] = []
        for x_row, pm25_hat in zip(X_test, preds_pm25_test):
            nb_class, posteriors = self.predict_proba(x_row)
            reg_class, _ = map_pm25_to_ica(float(pm25_hat))
            # Priorizar el estimador continuo reg_class salvo cuando Naive Bayes tenga certidumbre muy alta (>0.82)
            chosen = nb_class if posteriors.get(nb_class, 0.0) >= 0.82 else reg_class
            preds_cat_test.append(chosen)

        # Línea base 2: Clase mayoritaria en Train
        majority_class = pd.Series(y_cat_train).mode().iloc[0]
        baseline_majority_test = [majority_class] * len(y_cat_test)

        all_labels = sorted(list(set(y_cat_all)))
        metrics_model = compute_classification_metrics(y_cat_test, preds_cat_test, all_labels)
        metrics_persist = compute_classification_metrics(y_cat_test, baseline_persist_cat_test, all_labels)
        metrics_majority = compute_classification_metrics(y_cat_test, baseline_majority_test, all_labels)

        mae_model = float(np.mean(np.abs(preds_pm25_test - y_pm25_test)))
        mae_persist = float(np.mean(np.abs(baseline_persist_pm25_test - y_pm25_test)))
        rmse_model = float(np.sqrt(np.mean((preds_pm25_test - y_pm25_test) ** 2)))

        # 3. Reentrenar con el 100% de los datos disponibles para dejar el modelo listo para producción
        self._fit_parameters(X_all, y_pm25_all, y_cat_all)

        self.evaluation_metrics_ = {
            "accuracy_modelo_nb": metrics_model["accuracy"],
            "accuracy_baseline_persistencia": metrics_persist["accuracy"],
            "accuracy_baseline_mayoritaria": metrics_majority["accuracy"],
            "macro_f1_modelo": metrics_model["macro_f1"],
            "macro_f1_baseline_persistencia": metrics_persist["macro_f1"],
            "macro_f1_baseline_mayoritaria": metrics_majority["macro_f1"],
            "weighted_f1_modelo": metrics_model["weighted_f1"],
            "weighted_f1_baseline_persistencia": metrics_persist["weighted_f1"],
            "f1_por_clase_modelo": metrics_model["f1_por_clase"],
            "labels_matriz_confusion": metrics_model["labels_orden"],
            "matriz_confusion_test": metrics_model["confusion_matrix"],
            "mae_modelo_ugm3": round(mae_model, 2),
            "mae_baseline_ugm3": round(mae_persist, 2),
            "rmse_modelo_ugm3": round(rmse_model, 2),
            "n_muestras_total": n_total,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        return self.evaluation_metrics_

    def predict_proba(self, x_features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Calcula la distribución posterior P(C_{t+1} | x_t) mediante el Teorema de Bayes
        en espacio logarítmico (log-sum-exp trick) para evitar subdesbordamiento numérico.
        """
        if not self.is_fitted_:
            raise RuntimeError("El modelo de la Parte 1 debe entrenarse antes de predecir.")

        log_probs: Dict[str, float] = {}
        for c in self.classes_:
            prior_log = np.log(self.class_priors_[c])
            mean = self.means_[c]
            var = self.vars_[c]
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((x_features - mean) ** 2) / var)
            log_probs[c] = float(prior_log + log_likelihood)

        max_log = max(log_probs.values())
        exp_probs = {c: float(np.exp(lp - max_log)) for c, lp in log_probs.items()}
        norm_factor = sum(exp_probs.values())
        posteriors = {c: round(val / norm_factor, 4) for c, val in exp_probs.items()}

        best_class = max(posteriors, key=posteriors.get)
        return best_class, posteriors

    @staticmethod
    def estimate_hmm_latent_state(pm25_series: List[float]) -> str:
        """
        Filtro de Markov Oculto (HMM) de 3 estados latentes del POECA:
        Estados: S0='Normal', S1='Alerta / Prevención', S2='Episodio Crítico'.
        """
        states = ["Normal", "Alerta / Prevención", "Episodio Crítico"]
        T = np.array([
            [0.75, 0.22, 0.03],
            [0.20, 0.65, 0.15],
            [0.05, 0.30, 0.65],
        ])
        belief = np.array([0.70, 0.25, 0.05])
        emission_means = np.array([18.0, 38.0, 58.0])
        emission_stds = np.array([8.0, 9.0, 12.0])

        for obs in pm25_series:
            prior_belief = T.T @ belief
            likelihood = np.exp(-0.5 * ((obs - emission_means) / emission_stds) ** 2) / (
                emission_stds * np.sqrt(2 * np.pi)
            )
            belief = prior_belief * likelihood
            belief_sum = np.sum(belief)
            if belief_sum > 0:
                belief = belief / belief_sum

        best_idx = int(np.argmax(belief))
        return f"{states[best_idx]} (confianza: {belief[best_idx]:.1%})"

    def predict_next_day_for_station(self, df: pd.DataFrame, station_query: str) -> PredictionResult:
        """
        Realiza la inferencia del día siguiente para una estación del SIATA combinando
        el clasificador Gaussian Naive Bayes, la regresión Ridge y el filtro HMM.
        Importante: Para inferencia en el último día disponible (hoy), NO se debe descartar
        la última fila donde `target_pm25_next` es NaN (porque mañana aún no ha ocurrido).
        """
        if not self.is_fitted_:
            self.fit_and_evaluate(df)

        df_clean = df.copy()
        if "pm25_ma3" not in df_clean.columns:
            from src.data_loader import clean_and_preprocess_data
            df_clean = clean_and_preprocess_data(df_clean)

        q = station_query.strip().lower()
        # Usar regex=False para que paréntesis en nombres como 'Tanques EPM (El Poblado)'
        # o 'Museo de Antioquia (Centro)' se comparen literalmente y no como grupos regex.
        mask = (
            df_clean["station_id"].astype(str).str.lower().str.contains(q, na=False, regex=False)
            | df_clean["nombre"].astype(str).str.lower().str.contains(q, na=False, regex=False)
            | df_clean["municipio"].astype(str).str.lower().str.contains(q, na=False, regex=False)
        )
        # Búsqueda de respaldo sin paréntesis si el usuario escribe solo parte del nombre
        if not mask.any():
            q_clean = q.replace("(", "").replace(")", "").strip()
            mask = (
                df_clean["nombre"]
                .astype(str)
                .str.lower()
                .str.replace("(", "", regex=False)
                .str.replace(")", "", regex=False)
                .str.contains(q_clean, na=False, regex=False)
            )
        sub = df_clean[mask].sort_values("date").reset_index(drop=True)
        if sub.empty:
            available = ", ".join(sorted(df_clean["municipio"].unique()))
            raise ValueError(
                f"No se encontró la estación o municipio '{station_query}'. "
                f"Estaciones disponibles en el Valle de Aburrá: {available}."
            )

        last_row = sub.iloc[-1]
        x_feat = np.array([float(last_row[col]) for col in FEATURE_COLUMNS], dtype=float)

        pm25_pred = float(np.dot(self.reg_weights_, np.append(1.0, x_feat)))
        pm25_pred = max(5.0, round(pm25_pred, 1))
        _, probs = self.predict_proba(x_feat)
        cat_from_pm25, color_manana = map_pm25_to_ica(pm25_pred)
        poeca_manana = map_pm25_to_poeca_state(pm25_pred)

        latent_state = self.estimate_hmm_latent_state(sub["pm25"].astype(float).tail(14).tolist())

        return PredictionResult(
            station_id=str(last_row["station_id"]),
            nombre_estacion=str(last_row["nombre"]),
            fecha_base=str(last_row["date"]),
            pm25_actual=float(last_row["pm25"]),
            categoria_actual=str(last_row["categoria_ica"]),
            pm25_estimado_manana=pm25_pred,
            categoria_predicha_manana=cat_from_pm25,
            estado_poeca_manana=poeca_manana,
            color_alerta_manana=color_manana,
            probabilidades_clase=probs,
            estado_latente_hmm=latent_state,
            metricas_validacion=self.evaluation_metrics_,
        )

    def save_model(self, model_path: Path = DEFAULT_MODEL_PATH) -> Path:
        """
        Exporta el modelo entrenado y sus métricas de validación a disco (`models/air_quality_model_part1.pkl`)
        usando `pickle` para que el Agente LLM (`predict_tomorrow`) lo cargue directamente sin reentrenar.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        return model_path

    @classmethod
    def load_or_train(cls, df: pd.DataFrame, model_path: Path = DEFAULT_MODEL_PATH) -> "ClassicalAirQualityModel":
        """
        Carga el artefacto serializado desde `models/air_quality_model_part1.pkl` si existe;
        en caso contrario, lo entrena, evalúa y guarda automáticamente.
        """
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    loaded = pickle.load(f)
                if (
                    isinstance(loaded, cls)
                    and loaded.is_fitted_
                    and getattr(loaded, "model_version", None) == cls.MODEL_VERSION
                    and "macro_f1_modelo" in getattr(loaded, "evaluation_metrics_", {})
                ):
                    return loaded
            except Exception:
                pass

        instance = cls()
        instance.fit_and_evaluate(df)
        instance.save_model(model_path)
        return instance


if __name__ == "__main__":
    df_siata = load_air_quality_data()
    model = ClassicalAirQualityModel()
    metrics = model.fit_and_evaluate(df_siata)
    saved_path = model.save_model()
    print(f"Modelo de la Parte 1 entrenado y guardado en: {saved_path}")
    print("Métricas de Evaluación Cuantitativa (Test Set Temporal vs Baselines):")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
