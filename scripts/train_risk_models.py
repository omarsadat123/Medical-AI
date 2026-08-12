"""
Train educational risk-estimation models.

Generates clinically plausible synthetic cohorts (with noise), trains
RandomForest + XGBoost classifiers, and writes artifacts to models/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.features import DEFAULT_MEDIANS, RISK_FEATURE_SPECS  # noqa: E402

MODELS_DIR = ROOT / "models"
RNG = np.random.default_rng(42)


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def synthesize(condition: str, n: int = 4000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic X, y for a condition using noisy clinical rules."""
    features = list(RISK_FEATURE_SPECS[condition]["features"])
    k = len(features)
    X = np.zeros((n, k), dtype=float)

    # Draw base population near healthy medians, then enrich high-risk cases
    for j, name in enumerate(features):
        base = DEFAULT_MEDIANS[name]
        noise = RNG.normal(0, abs(base) * 0.12 + 0.5, size=n)
        X[:, j] = base + noise

    # Enrich ~40% as elevated-risk profiles
    risk_idx = RNG.choice(n, size=int(n * 0.4), replace=False)

    if condition == "diabetes":
        fi = {f: i for i, f in enumerate(features)}
        X[risk_idx, fi["glucose_fasting"]] = RNG.normal(130, 20, size=len(risk_idx))
        X[risk_idx, fi["hba1c"]] = RNG.normal(6.5, 0.7, size=len(risk_idx))
        X[risk_idx, fi["bmi"]] = RNG.normal(31, 4, size=len(risk_idx))
        X[risk_idx, fi["triglycerides"]] = RNG.normal(190, 40, size=len(risk_idx))
        X[:, fi["glucose_fasting"]] = _clip(X[:, fi["glucose_fasting"]], 60, 300)
        X[:, fi["hba1c"]] = _clip(X[:, fi["hba1c"]], 4.0, 14.0)
        X[:, fi["bmi"]] = _clip(X[:, fi["bmi"]], 16, 50)
        X[:, fi["age"]] = _clip(X[:, fi["age"]], 18, 90)
        # Label from noisy clinical score
        score = (
            0.045 * (X[:, fi["glucose_fasting"]] - 90)
            + 2.2 * (X[:, fi["hba1c"]] - 5.4)
            + 0.08 * (X[:, fi["bmi"]] - 24)
            + 0.004 * (X[:, fi["triglycerides"]] - 120)
            + 0.01 * (X[:, fi["age"]] - 45)
            + RNG.normal(0, 0.6, size=n)
        )
        y = (score > 1.2).astype(int)

    elif condition == "heart":
        fi = {f: i for i, f in enumerate(features)}
        X[risk_idx, fi["ldl"]] = RNG.normal(160, 25, size=len(risk_idx))
        X[risk_idx, fi["total_cholesterol"]] = RNG.normal(240, 30, size=len(risk_idx))
        X[risk_idx, fi["hdl"]] = RNG.normal(38, 6, size=len(risk_idx))
        X[risk_idx, fi["systolic_bp"]] = RNG.normal(145, 12, size=len(risk_idx))
        X[:, fi["hdl"]] = _clip(X[:, fi["hdl"]], 20, 100)
        X[:, fi["ldl"]] = _clip(X[:, fi["ldl"]], 40, 250)
        X[:, fi["systolic_bp"]] = _clip(X[:, fi["systolic_bp"]], 90, 200)
        score = (
            0.02 * (X[:, fi["ldl"]] - 100)
            + 0.015 * (X[:, fi["total_cholesterol"]] - 180)
            - 0.05 * (X[:, fi["hdl"]] - 50)
            + 0.03 * (X[:, fi["systolic_bp"]] - 120)
            + 0.01 * (X[:, fi["triglycerides"]] - 120) / 10
            + 0.02 * (X[:, fi["age"]] - 45)
            + RNG.normal(0, 0.7, size=n)
        )
        y = (score > 1.0).astype(int)

    elif condition == "kidney":
        fi = {f: i for i, f in enumerate(features)}
        X[risk_idx, fi["creatinine"]] = RNG.normal(1.6, 0.35, size=len(risk_idx))
        X[risk_idx, fi["bun"]] = RNG.normal(28, 6, size=len(risk_idx))
        X[risk_idx, fi["egfr"]] = RNG.normal(55, 12, size=len(risk_idx))
        X[:, fi["creatinine"]] = _clip(X[:, fi["creatinine"]], 0.4, 4.0)
        X[:, fi["bun"]] = _clip(X[:, fi["bun"]], 5, 80)
        X[:, fi["egfr"]] = _clip(X[:, fi["egfr"]], 5, 130)
        score = (
            2.5 * (X[:, fi["creatinine"]] - 1.0)
            + 0.08 * (X[:, fi["bun"]] - 15)
            - 0.04 * (X[:, fi["egfr"]] - 90)
            + 0.015 * (X[:, fi["age"]] - 45)
            + RNG.normal(0, 0.5, size=n)
        )
        y = (score > 0.8).astype(int)

    elif condition == "anemia":
        fi = {f: i for i, f in enumerate(features)}
        X[risk_idx, fi["hemoglobin"]] = RNG.normal(10.2, 1.0, size=len(risk_idx))
        X[risk_idx, fi["hematocrit"]] = RNG.normal(32, 3, size=len(risk_idx))
        X[risk_idx, fi["iron"]] = RNG.normal(40, 10, size=len(risk_idx))
        X[risk_idx, fi["ferritin"]] = RNG.normal(18, 8, size=len(risk_idx))
        X[risk_idx, fi["sex_female"]] = (RNG.random(len(risk_idx)) > 0.35).astype(float)
        X[:, fi["hemoglobin"]] = _clip(X[:, fi["hemoglobin"]], 6, 18)
        X[:, fi["hematocrit"]] = _clip(X[:, fi["hematocrit"]], 20, 55)
        X[:, fi["sex_female"]] = _clip(X[:, fi["sex_female"]], 0, 1)
        # Sex-aware threshold noise
        hb_thresh = np.where(X[:, fi["sex_female"]] > 0.5, 12.0, 13.0)
        score = (
            1.4 * (hb_thresh - X[:, fi["hemoglobin"]])
            + 0.08 * (36 - X[:, fi["hematocrit"]])
            + 0.03 * (60 - X[:, fi["iron"]])
            + 0.02 * (30 - X[:, fi["ferritin"]])
            + RNG.normal(0, 0.5, size=n)
        )
        y = (score > 0.9).astype(int)
    else:
        raise ValueError(condition)

    return X, y, features


def train_one(condition: str) -> dict:
    X, y, features = synthesize(condition)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)

    def metrics(model) -> dict:
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        }

    rf_m = metrics(rf)
    xgb_m = metrics(xgb)
    # Prefer the model with better ROC-AUC
    if xgb_m["roc_auc"] >= rf_m["roc_auc"]:
        chosen_name, chosen, chosen_m = "xgboost", xgb, xgb_m
    else:
        chosen_name, chosen, chosen_m = "random_forest", rf, rf_m

    medians = {f: float(np.median(X_train[:, i])) for i, f in enumerate(features)}
    importances = {
        f: float(v) for f, v in zip(features, chosen.feature_importances_, strict=True)
    }

    artifact = {
        "condition": condition,
        "label": RISK_FEATURE_SPECS[condition]["label"],
        "description": RISK_FEATURE_SPECS[condition]["description"],
        "features": features,
        "medians": medians,
        "model_type": chosen_name,
        "metrics": {"random_forest": rf_m, "xgboost": xgb_m, "selected": chosen_m},
        "feature_importances": importances,
        "disclaimer": (
            "Educational risk estimate only — not a diagnosis or clinical decision tool."
        ),
    }

    out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(chosen, out_dir / f"{condition}_model.joblib")
    (out_dir / f"{condition}_meta.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for condition in RISK_FEATURE_SPECS:
        print(f"Training {condition}...")
        art = train_one(condition)
        sel = art["metrics"]["selected"]
        print(
            f"  model={art['model_type']}  "
            f"auc={sel['roc_auc']:.3f}  f1={sel['f1']:.3f}  acc={sel['accuracy']:.3f}"
        )
        summary[condition] = {
            "model_type": art["model_type"],
            "metrics": art["metrics"]["selected"],
        }
    (MODELS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
