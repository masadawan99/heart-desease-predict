"""
Shared training logic for the Heart Disease Prediction project.

Both train_model.py and Heart_Disease_Prediction.ipynb import this module so
the preprocessing and model are defined in exactly one place.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).parent
CLEVELAND_PATH = BASE_DIR / "processed.cleveland.data"
KAGGLE_PATH = BASE_DIR / "data" / "kaggle_heart_failure" / "heart.csv"
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"

# ---- Kaggle (fedesoriano) schema -------------------------------------------
KAGGLE_TARGET = "HeartDisease"          # 0 = no heart disease, 1 = heart disease
NUMERIC_FEATURES = ["Age", "MaxHR"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType"]
FEATURES = ["Age", "Sex", "ChestPainType", "MaxHR"]   # exactly four, in this order
SEX_CATEGORIES = ["M", "F"]
CHEST_PAIN_CATEGORIES = ["TA", "ATA", "NAP", "ASY"]

# ---- Cleveland (baseline) schema -------------------------------------------
CLEVELAND_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
CLEVELAND_FEATURES = ["age", "sex", "cp", "thalach"]

SPLIT = dict(test_size=0.2, random_state=0)   # 80/20, stratified, seeded


def load_kaggle(path=KAGGLE_PATH):
    """Load the raw Kaggle CSV and validate the columns we need."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [KAGGLE_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def clean_kaggle(df):
    """Return a cleaned working copy (the raw data is never modified).

    Only rows with missing or invalid values in the four predictors / target are
    dropped. Duplicates are NOT removed (see notebook: none exist in this file,
    and identical predictor values across different patients are legitimate).
    """
    work = df[FEATURES + [KAGGLE_TARGET]].copy()
    work = work.dropna()
    work = work[work["Sex"].isin(SEX_CATEGORIES)
                & work["ChestPainType"].isin(CHEST_PAIN_CATEGORIES)
                & work["Age"].between(1, 120)
                & work["MaxHR"].between(40, 250)
                & work[KAGGLE_TARGET].isin([0, 1])]
    return work.reset_index(drop=True)


def split_kaggle(work):
    """Return X_train, X_test, y_train, y_test with exactly four predictors."""
    X, y = work[FEATURES], work[KAGGLE_TARGET]
    assert list(X.columns) == FEATURES and KAGGLE_TARGET not in X.columns
    return train_test_split(X, y, stratify=y, **SPLIT)


def build_pipeline(estimator=None):
    """Preprocessing + classifier. Encoders/scalers are fitted on training data only."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(categories=[SEX_CATEGORIES, CHEST_PAIN_CATEGORIES],
                              handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    if estimator is None:
        estimator = SVC(gamma="auto", random_state=0)
    return Pipeline([("preprocess", pre), ("model", estimator)])


def evaluate(model, X_test, y_test):
    """Held-out metrics. ROC-AUC uses decision scores (ranking only, not probabilities)."""
    pred = model.predict(X_test)
    out = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    if y_test.nunique() == 2:
        score = (model.decision_function(X_test) if hasattr(model, "decision_function")
                 else model.predict_proba(X_test)[:, 1])
        out["roc_auc"] = roc_auc_score(y_test, score)
    return out


def run_cleveland_baseline(path=CLEVELAND_PATH):
    """Experiment A: the original project's workflow, unchanged (raw integer
    features, SVC(gamma='auto'), stratified 80/20 split with random_state=0)."""
    df = pd.read_csv(path, header=None, names=CLEVELAND_COLUMNS, na_values="?")
    df = df.dropna().reset_index(drop=True)
    y = (df["num"] > 0).astype(int)
    X = df[CLEVELAND_FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, **SPLIT)
    model = SVC(gamma="auto", random_state=0).fit(X_tr, y_tr)
    res = evaluate(model, X_te, y_te)
    res.update(n_rows=len(df), n_train=len(X_tr), n_test=len(X_te))
    return res
