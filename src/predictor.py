"""
predictor.py
------------
Thin inference wrapper used by the Streamlit dashboard.
Loads the trained pipeline once and exposes a predict() function.

Usage (from the dashboard):
    from src.predictor import Predictor
    p = Predictor()
    value_eur = p.predict({"age": 24, "position": 2, ...})
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent

_PIPELINE_PATH  = ROOT / "models" / "rf_pipeline.joblib"
_META_PATH      = ROOT / "models" / "feature_meta.json"
_ENCODINGS_PATH = ROOT / "data" / "processed" / "label_encodings.json"


class ModelNotTrainedError(RuntimeError):
    pass


class Predictor:
    """Loads the trained RF pipeline and runs inference."""

    def __init__(self) -> None:
        self._pipeline   = None
        self._feature_cols: list[str] = []
        self._log_transform: bool = True
        self._encodings: dict[str, dict] = {}
        self._loaded = False
        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        missing = [p for p in [_PIPELINE_PATH, _META_PATH] if not p.exists()]
        if missing:
            raise ModelNotTrainedError(
                f"Model files not found: {missing}\n"
                "Run  python src/train.py  first."
            )

        self._pipeline = joblib.load(_PIPELINE_PATH)

        with open(_META_PATH) as f:
            meta = json.load(f)
        self._feature_cols  = meta["feature_columns"]
        self._log_transform = meta.get("log_transform", True)

        if _ENCODINGS_PATH.exists():
            with open(_ENCODINGS_PATH) as f:
                self._encodings = json.load(f)

        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_cols)

    @property
    def encodings(self) -> dict[str, dict]:
        """Return {column_name: {label: int_code}} mappings."""
        return dict(self._encodings)

    def encode_label(self, column: str, label: str) -> int:
        """
        Convert a human-readable categorical label to its integer code.
        Returns -1 for unknown labels (handled gracefully by the model).
        """
        return self._encodings.get(column, {}).get(label, -1)

    def decode_label(self, column: str, code: int) -> str:
        """Reverse a label encoding code back to a human-readable string."""
        mapping = self._encodings.get(column, {})
        reverse = {v: k for k, v in mapping.items()}
        return reverse.get(code, "Unknown")

    def predict(self, features: dict[str, Any]) -> float:
        """
        Predict a player's market value in EUR.

        Parameters
        ----------
        features : dict
            Keys must match feature_columns (integer codes for categoricals).
            Missing keys default to 0.

        Returns
        -------
        float
            Estimated market value in EUR (≥ 0).
        """
        if not self._loaded:
            raise ModelNotTrainedError("Model not loaded.")

        # Build a single-row DataFrame in the exact column order used during training
        row = {col: features.get(col, 0) for col in self._feature_cols}
        df  = pd.DataFrame([row])

        log_pred = self._pipeline.predict(df)[0]

        if self._log_transform:
            value = float(np.expm1(log_pred))
        else:
            value = float(log_pred)

        return max(value, 0.0)   # predictions cannot be negative

    def predict_batch(self, records: list[dict[str, Any]]) -> list[float]:
        """Predict values for a list of player feature dicts."""
        return [self.predict(r) for r in records]

    def is_ready(self) -> bool:
        return self._loaded
