"""
Heart Disease Prediction System - Training Script (Kaggle dataset)
==================================================================
Trains an SVC on the Kaggle "Heart Failure Prediction" dataset using exactly
four predictors (Age, Sex, ChestPainType, MaxHR), evaluates on held-out data,
and saves the complete fitted pipeline to heart_disease_model.pkl.

Usage:  python train_model.py
Needs:  data/kaggle_heart_failure/heart.csv  (see README for download steps)
"""
import json
import sys

import joblib
import pandas as pd
import sklearn

import heart_pipeline as hp


def main():
    if not hp.KAGGLE_PATH.exists():
        sys.exit(f"Dataset not found: {hp.KAGGLE_PATH}\nSee README.md for download instructions.")

    raw = hp.load_kaggle()
    print(f"Raw Kaggle data: {raw.shape}")
    work = hp.clean_kaggle(raw)
    print(f"After cleaning:  {work.shape} (dropped {len(raw) - len(work)} rows)")
    print(f"Target '{hp.KAGGLE_TARGET}' distribution:\n{work[hp.KAGGLE_TARGET].value_counts().sort_index()}")

    X_train, X_test, y_train, y_test = hp.split_kaggle(work)
    print(f"\nPredictors ({len(X_train.columns)}): {list(X_train.columns)}")
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    model = hp.build_pipeline().fit(X_train, y_train)
    metrics = hp.evaluate(model, X_test, y_test)
    print("\nHeld-out test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    artifact = {
        "pipeline": model,
        "features": hp.FEATURES,
        "target": hp.KAGGLE_TARGET,
        "target_definition": {0: "No heart disease", 1: "Heart disease present"},
        "categories": {"Sex": hp.SEX_CATEGORIES, "ChestPainType": hp.CHEST_PAIN_CATEGORIES},
        "dataset": "fedesoriano/heart-failure-prediction (heart.csv)",
        "sklearn_version": sklearn.__version__,
        "test_metrics": metrics,
    }
    tmp = hp.MODEL_PATH.with_suffix(".tmp")
    joblib.dump(artifact, tmp)

    # Reload test before replacing the existing model file
    loaded = joblib.load(tmp)
    again = hp.evaluate(loaded["pipeline"], X_test, y_test)
    assert again == metrics, "Reloaded model gives different metrics!"
    sample = pd.DataFrame([{"Age": 54, "Sex": "M", "ChestPainType": "ASY", "MaxHR": 120}])[loaded["features"]]
    print(f"\nReload OK. Sample prediction {sample.iloc[0].to_dict()} -> {loaded['pipeline'].predict(sample)[0]}")
    tmp.replace(hp.MODEL_PATH)
    print(f"Saved: {hp.MODEL_PATH.name}")

    with open(hp.BASE_DIR / "metrics.json", "w") as f:
        json.dump({"kaggle": metrics, "n_train": len(X_train), "n_test": len(X_test),
                   "features": hp.FEATURES}, f, indent=2)


if __name__ == "__main__":
    main()
