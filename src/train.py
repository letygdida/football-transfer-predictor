"""
train.py
--------
Trains and evaluates two models:
  1. Baseline — Median absolute value (predict the training-set median for
     every player; a realistic floor to beat).
  2. Random Forest Regressor — the primary production model.

Evaluation metrics (on held-out test set):
  - MAE  : Mean Absolute Error in EUR
  - RMSE : Root Mean Squared Error in EUR
  - R²   : Coefficient of determination (1.0 = perfect)

Outputs:
  models/rf_pipeline.joblib   <- trained pipeline (scaler + RF)
  models/metrics.json         <- test-set evaluation results
  models/feature_importance.csv

Run:
  python src/train.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"

FEATURES_PATH = PROCESSED / "features.csv"
ENCODINGS_PATH = PROCESSED / "label_encodings.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():  # -> tuple[pd.DataFrame, list[str]]
    if not FEATURES_PATH.exists():
        print("  ✗ features.csv not found.  Run: python src/preprocess.py")
        sys.exit(1)

    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    print(f"  Loaded features.csv  shape={df.shape}")

    # Separate target
    TARGET = "market_value_in_eur"
    feature_cols = [c for c in df.columns if c != TARGET and c != "player_id"]

    # Sanity check — no leakage columns sneaked in
    LEAKAGE_GUARD = ["highest_market_value", "transfer_fee"]
    for col in LEAKAGE_GUARD:
        if any(col in c for c in feature_cols):
            print(f"  ✗ Potential leakage column detected: {col}")
            sys.exit(1)

    return df, feature_cols


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  ── {name} ──")
    print(f"     MAE  : €{mae:>15,.0f}")
    print(f"     RMSE : €{rmse:>15,.0f}")
    print(f"     R²   :  {r2:>14.4f}")

    return {"model": name, "MAE_EUR": round(mae, 2),
            "RMSE_EUR": round(rmse, 2), "R2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Football Transfer Value — Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading processed features ...")
    df, feature_cols = load_data()

    TARGET = "market_value_in_eur"
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Log-transform the target: market values are highly right-skewed.
    # We train on log(value) and exponentiate predictions back.
    y_log = np.log1p(y)

    print(f"  Features used: {len(feature_cols)}")
    print(f"  Target range:  €{y.min():,.0f}  –  €{y.max():,.0f}")
    print(f"  Median value:  €{y.median():,.0f}")

    # ------------------------------------------------------------------
    # 2. Train/test split  (80/20, stratified by log-value quartile)
    # ------------------------------------------------------------------
    print("\n[2/5] Splitting data  (80 % train / 20 % test) ...")
    quartile_bins = pd.qcut(y_log, q=4, labels=False)
    X_train, X_test, y_log_train, y_log_test = train_test_split(
        X, y_log, test_size=0.20, random_state=42, stratify=quartile_bins
    )
    print(f"  Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")

    # Original-scale test targets (for human-readable metrics)
    y_test_eur = np.expm1(y_log_test)

    # ------------------------------------------------------------------
    # 3. Baseline model — median predictor
    # ------------------------------------------------------------------
    print("\n[3/5] Training baseline (median predictor) ...")
    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_log_train)
    baseline_pred_log = baseline.predict(X_test)
    baseline_pred_eur = np.expm1(baseline_pred_log)
    metrics_baseline  = evaluate("Baseline (Median)", y_test_eur.values, baseline_pred_eur)

    # ------------------------------------------------------------------
    # 4. Random Forest pipeline
    # ------------------------------------------------------------------
    print("\n[4/5] Training Random Forest ...")

    # Build pipeline: impute any residual NaNs, scale, then RF
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),          # centres features (helps RF less, but keeps the pipeline reusable)
        ("rf", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    rf_pipeline.fit(X_train, y_log_train)
    rf_pred_log = rf_pipeline.predict(X_test)
    rf_pred_eur = np.expm1(rf_pred_log)
    metrics_rf  = evaluate("Random Forest", y_test_eur.values, rf_pred_eur)

    # 5-fold cross-val R² on training set
    print("\n  Cross-validation R² (5-fold, training data) ...")
    cv_scores = cross_val_score(rf_pipeline, X_train, y_log_train,
                                cv=5, scoring="r2", n_jobs=-1)
    print(f"  CV R²:  {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")
    metrics_rf["CV_R2_mean"] = round(float(cv_scores.mean()), 4)
    metrics_rf["CV_R2_std"]  = round(float(cv_scores.std()),  4)

    # ------------------------------------------------------------------
    # 5. Save everything
    # ------------------------------------------------------------------
    print("\n[5/5] Saving models and metrics ...")

    # Save RF pipeline
    rf_path = MODELS_DIR / "rf_pipeline.joblib"
    joblib.dump(rf_pipeline, rf_path)
    print(f"  ✓ Saved RF pipeline  →  {rf_path}")

    # Save feature column order (the dashboard needs this to build inputs in
    # the exact same order the model was trained on)
    feature_meta = {
        "feature_columns": feature_cols,
        "target":          TARGET,
        "log_transform":   True,
    }
    meta_path = MODELS_DIR / "feature_meta.json"
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"  ✓ Saved feature meta →  {meta_path}")

    # Save metrics
    all_metrics = [metrics_baseline, metrics_rf]
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✓ Saved metrics      →  {metrics_path}")

    # Save feature importances
    rf_model = rf_pipeline.named_steps["rf"]
    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_path = MODELS_DIR / "feature_importance.csv"
    importances.to_csv(fi_path, index=False)
    print(f"  ✓ Saved importances  →  {fi_path}")

    print()
    print("  Top 10 most important features:")
    for _, row in importances.head(10).iterrows():
        bar_len = int(row["importance"] * 400)
        print(f"    {row['feature']:<35} {'█' * bar_len} {row['importance']:.4f}")

    print()
    print("  Summary")
    print(f"    Baseline  MAE  : €{metrics_baseline['MAE_EUR']:>14,.0f}")
    print(f"    RF        MAE  : €{metrics_rf['MAE_EUR']:>14,.0f}")
    improvement = (1 - metrics_rf["MAE_EUR"] / metrics_baseline["MAE_EUR"]) * 100
    print(f"    MAE improvement over baseline: {improvement:.1f}%")
    print()
    print("  Next step: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
