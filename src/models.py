from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

@dataclass
class ModelSpec:
    task: str
    pipeline: Pipeline
    name: str

def regression_models(numeric_cols) -> list[ModelSpec]:
    scaler = ColumnTransformer([("num", StandardScaler(), numeric_cols)], remainder="drop")
    ridge = Pipeline([("prep", scaler), ("model", Ridge(alpha=1.0))])
    rf = Pipeline([("model", RandomForestRegressor(n_estimators=300, random_state=42))])
    return [
        ModelSpec("regression", ridge, "Ridge"),
        ModelSpec("regression", rf, "RandomForestRegressor"),
    ]

def classification_models(numeric_cols) -> list[ModelSpec]:
    scaler = ColumnTransformer([("num", StandardScaler(), numeric_cols)], remainder="drop")
    logit = Pipeline([("prep", scaler), ("model", LogisticRegression(max_iter=200))])
    rf = Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=42))])
    return [
        ModelSpec("classification", logit, "LogisticRegression"),
        ModelSpec("classification", rf, "RandomForestClassifier"),
    ]
