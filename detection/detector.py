from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap
from sklearn.ensemble import IsolationForest

from config import CONTAMINATION, SHAP_TOP_N, WARMUP_SIZE
from processing.feature_engine import FEATURE_NAMES


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float
    top_contributors: list[tuple[str, float]]


class AnomalyDetector:
    def __init__(
        self,
        contamination: float = CONTAMINATION,
        warmup_size: int = WARMUP_SIZE,
        top_n: int = SHAP_TOP_N,
    ) -> None:
        self._contamination = contamination
        self._warmup_size = warmup_size
        self._top_n = top_n
        self._buffer: list[np.ndarray] = []
        self._model: IsolationForest | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._fitted = False

    @property
    def is_ready(self) -> bool:
        return self._fitted

    @property
    def warmup_progress(self) -> tuple[int, int]:
        return len(self._buffer), self._warmup_size

    def _fit(self) -> None:
        X = np.vstack(self._buffer)
        self._model = IsolationForest(
            contamination=self._contamination,
            n_estimators=200,
            max_samples="auto",
            random_state=42,
        )
        self._model.fit(X)
        self._explainer = shap.TreeExplainer(self._model)
        self._fitted = True

    def update(self, feature_vector: np.ndarray) -> AnomalyResult | None:
        if not self._fitted:
            self._buffer.append(feature_vector)
            if len(self._buffer) >= self._warmup_size:
                self._fit()
            return None

        return self._score(feature_vector)

    def _score(self, feature_vector: np.ndarray) -> AnomalyResult:
        assert self._model is not None
        assert self._explainer is not None

        x = feature_vector.reshape(1, -1)
        label = int(self._model.predict(x)[0])        # 1 = inlier, -1 = outlier
        score = float(self._model.score_samples(x)[0])  # lower = more anomalous

        shap_vals: np.ndarray = self._explainer.shap_values(x)
        contributions = list(zip(FEATURE_NAMES, shap_vals[0].tolist()))
        top = sorted(contributions, key=lambda t: abs(t[1]), reverse=True)[: self._top_n]

        return AnomalyResult(
            is_anomaly=(label == -1),
            score=round(score, 5),
            top_contributors=top,
        )
