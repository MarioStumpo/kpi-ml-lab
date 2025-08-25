from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

# ---------------- Label rules (used for classification) ----------------
def add_label_rule(df: pd.DataFrame, rule: str) -> pd.Series:
    """
    Built-in labeling utilities for didactic classification tasks.
    - high_sprinter: label=1 if sprints_per90 above median
    - high_distance: label=1 if distance_per90_km above median
    """
    r = (rule or "").strip().lower()
    if r == "high_sprinter":
        if "sprints_per90" not in df.columns:
            raise ValueError("Missing column: sprints_per90 for label-rule high_sprinter")
        thr = df["sprints_per90"].median()
        return (df["sprints_per90"] > thr).astype(int)

    if r == "high_distance":
        if "distance_per90_km" not in df.columns:
            raise ValueError("Missing column: distance_per90_km for label-rule high_distance")
        thr = df["distance_per90_km"].median()
        return (df["distance_per90_km"] > thr).astype(int)

    raise ValueError("Unknown --label-rule. Try: high_sprinter | high_distance")

# ---------------- Default feature set ----------------
DEFAULT_FEATURES = [
    "minutes","distance_km","sprints","passes","passes_completed",
    "pass_accuracy_%","distance_per90_km","sprints_per90"
]

# ---------------- Leakage guardrails ----------------
def _regression_forbids(target: str) -> Dict[str, str]:
    """Columns that leak the regression target, with reasons."""
    t = (target or "").strip().lower()
    rules: Dict[str, str] = {}

    # Distance
    if t == "distance_km":
        rules["distance_per90_km"] = "derived from distance_km via minutes (distance*90/minutes)"
    if t == "distance_per90_km":
        rules["distance_km"] = "directly proportional to target"
        rules["minutes"] = "target is distance normalized by minutes"

    # Sprints
    if t == "sprints_per90":
        rules["sprints"] = "sprints_per90 = sprints*90/minutes"
        rules["minutes"] = "used to normalize sprints"

    # Passing accuracy
    if t == "pass_accuracy_%":
        rules["passes_completed"] = "accuracy ~ passes_completed / passes"
        rules["passes"] = "accuracy ~ passes_completed / passes"

    return rules

def _classification_forbids(label_rule: str | None) -> Dict[str, str]:
    """Columns that leak the label by definition."""
    rules: Dict[str, str] = {}
    lr = (label_rule or "").strip().lower()

    if lr == "high_sprinter":
        rules["sprints_per90"] = "label defined by sprints_per90 (median split)"
        rules["sprints"] = "with minutes reconstructs sprints_per90"
        rules["minutes"] = "used to normalize sprints to per90"

    if lr == "high_distance":
        rules["distance_per90_km"] = "label defined by distance_per90_km (median split)"
        rules["distance_km"] = "with minutes reconstructs distance_per90_km"
        rules["minutes"] = "used to normalize distance to per90"

    return rules

def leakage_forbids(task: str, target: str | None, label_rule: str | None) -> Dict[str, str]:
    if task == "regression" and target:
        return _regression_forbids(target)
    if task == "classification":
        return _classification_forbids(label_rule)
    return {}

# ---------------- Feature selection helpers ----------------
def sanitize_feature_list(df: pd.DataFrame, feats: List[str]) -> List[str]:
    """Keep only columns that exist and are not 'player'."""
    cols = set(df.columns)
    out: List[str] = []
    for f in feats:
        f2 = f.strip()
        if f2 and f2 != "player" and f2 in cols:
            out.append(f2)
    return out

def apply_exclusions(feats: List[str], forbids: Dict[str, str], extra_exclude: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Remove forbidden features and user excludes; return final list and removal reasons."""
    reasons = dict(forbids)
    exclude_set = set([f.strip() for f in extra_exclude if f.strip()])
    final: List[str] = []
    for f in feats:
        if f in reasons:
            continue
        if f in exclude_set:
            reasons[f] = "manually excluded via --exclude-features"
            continue
        final.append(f)
    return final, reasons

def make_features(df: pd.DataFrame,
                  task: str,
                  target: str | None = None,
                  label_rule: str | None = None,
                  include_features: List[str] | None = None,
                  exclude_features: List[str] | None = None) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Returns (X, used_features, removed_with_reasons)
    - Enforces leakage forbids automatically based on task/target/label_rule.
    - Applies optional include/exclude from CLI.
    """
    include_features = include_features or [c for c in DEFAULT_FEATURES if c in df.columns]
    exclude_features = exclude_features or []

    include_features = sanitize_feature_list(df, include_features)

    # never include 'player' or the target itself
    if target and target in include_features:
        include_features = [f for f in include_features if f != target]

    forbids = leakage_forbids(task, target, label_rule)
    used, removed = apply_exclusions(include_features, forbids, exclude_features)
    X = df[used].copy()
    return X, used, removed