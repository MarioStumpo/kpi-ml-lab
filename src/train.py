from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.cluster import KMeans

from .data import load_table
from .features import make_features
from .models import regression_models, classification_models
from .metrics import regression_report, classification_report, pretty, cv_summary
from .plotting import save_scatter, save_residuals, save_feature_importance, save_confusion_matrix

def _ensure_outdir(p: str | Path) -> Path:
    out = Path(p); out.mkdir(parents=True, exist_ok=True); return out

# ---------------- REGRESSION ----------------
def run_regression(df: pd.DataFrame, target: str, outdir: Path, seed: int, test_size: float,
                   include_features, exclude_features):
    # Build features with safety net
    X, used_feats, removed = make_features(
        df, task="regression", target=target,
        include_features=include_features, exclude_features=exclude_features
    )
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    y = df[target].values

    # Split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    # CV
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    for spec in regression_models(X.columns):
        cv_mae = -cross_val_score(spec.pipeline, X, y, scoring="neg_mean_absolute_error", cv=cv)
        cv_r2  =  cross_val_score(spec.pipeline, X, y, scoring="r2", cv=cv)
        spec.pipeline.fit(Xtr, ytr)
        pred = spec.pipeline.predict(Xte)
        rep = regression_report(yte, pred)
        results.append({
            "name": spec.name, "holdout": rep, "cv_mae": cv_mae, "cv_r2": cv_r2, "pipeline": spec.pipeline
        })

    best = sorted(results, key=lambda r: r["holdout"]["MAE"])[0]
    pipe = best["pipeline"]
    joblib.dump(pipe, outdir / "model.joblib")

    # Report
    mae_mean, mae_std = cv_summary(best["cv_mae"])
    r2_mean,  r2_std  = cv_summary(best["cv_r2"])
    lines = [
        "Task: Regression",
        f"Target: {target}",
        f"Best model: {best['name']}",
        "",
        "[Holdout metrics]",
        pretty(best["holdout"]),
        "",
        "[5-fold CV metrics] (computed on full X,y)",
        f"MAE (mean±std): {mae_mean:.4f} ± {mae_std:.4f}",
        f"R2  (mean±std): {r2_mean:.4f} ± {r2_std:.4f}",
        "",
        "[Used features]",
        ", ".join(used_feats) if used_feats else "(none)",
    ]
    if removed:
        lines += ["", "[Safety exclusions to prevent leakage]"]
        for k, v in removed.items():
            lines.append(f"- {k}: {v}")
    lines += ["", "[All models summary]"]
    for r in results:
        m_mean, m_std = cv_summary(r["cv_mae"])
        r_mean, r_std = cv_summary(r["cv_r2"])
        lines += [
            f"- {r['name']}:",
            f"    Holdout -> MAE={r['holdout']['MAE']:.4f}, R2={r['holdout']['R2']:.4f}",
            f"    CV      -> MAE={m_mean:.4f}±{m_std:.4f}, R2={r_mean:.4f}±{r_std:.4f}",
        ]
    (outdir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Artifacts
    y_pred = pipe.predict(Xte)
    save_residuals(yte, y_pred, outdir / "residuals.png")
    if hasattr(pipe[-1], "feature_importances_"):
        save_feature_importance(pipe[-1].feature_importances_, X.columns.to_list(), outdir / "feature_importance.png")
    pd.DataFrame({"y_true": yte, "y_pred": y_pred}).to_csv(outdir / "predictions.csv", index=False)

# ---------------- CLASSIFICATION ----------------
def run_classification(df: pd.DataFrame, label_rule: str, outdir: Path, seed: int, test_size: float,
                       include_features, exclude_features):
    # Create label
    from .features import _classification_forbids  # just for readable label name
    y_name = f"label:{label_rule}"
    # Build features with safety net
    X, used_feats, removed = make_features(
        df, task="classification", label_rule=label_rule,
        include_features=include_features, exclude_features=exclude_features
    )

    # Build label values (after feature safety calc to avoid circular)
    from .features import add_label_rule
    y = add_label_rule(df, label_rule)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    results = []
    for spec in classification_models(X.columns):
        spec.pipeline.fit(Xtr, ytr)
        pred = spec.pipeline.predict(Xte)
        rep = classification_report(yte, pred)
        results.append((spec.name, rep, pred, spec.pipeline))

    name, rep, pred_best, pipe = sorted(results, key=lambda x: -x[1]["F1"])[0]
    joblib.dump(pipe, outdir / "model.joblib")

    lines = [
        "Task: Classification",
        f"Target: {y_name}",
        f"Model: {name}",
        pretty(rep),
        "",
        "[Used features]",
        ", ".join(used_feats) if used_feats else "(none)",
    ]
    if removed:
        lines += ["", "[Safety exclusions to prevent leakage]"]
        for k, v in removed.items():
            lines.append(f"- {k}: {v}")
    (outdir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cols = [c for c in ["distance_per90_km", "sprints_per90"] if c in X.columns]
    if len(cols) == 2:
        tmp = Xte.copy(); tmp[y_name] = yte.values
        save_scatter(tmp, cols[0], cols[1], outdir / "scatter.png", hue=y_name)
    save_confusion_matrix(yte, pred_best, outdir / "confusion_matrix.png", normalize="true")

    pd.DataFrame({"y_true": yte, "y_pred": pred_best}).to_csv(outdir / "predictions.csv", index=False)

# ---------------- CLUSTERING ----------------
def run_clustering(df: pd.DataFrame, features: list[str], clusters: int, outdir: Path, seed: int):
    feats = features or [c for c in ["distance_per90_km","sprints_per90"] if c in df.columns]
    X = df[feats].copy()
    km = KMeans(n_clusters=clusters, n_init="auto", random_state=seed)
    labels = km.fit_predict(X)
    out = (df[["player"] + feats] if "player" in df.columns else df[feats]).copy()
    out["cluster"] = labels
    out.to_csv(outdir / "clusters.csv", index=False)
    if len(feats) == 2:
        save_scatter(out, feats[0], feats[1], outdir / "scatter.png", hue="cluster")
    (outdir / "report.txt").write_text(
        f"Task: Clustering\nAlgorithm: KMeans k={clusters}\nFeatures: {feats}\nCenters:\n{km.cluster_centers_}\n",
        encoding="utf-8"
    )
    joblib.dump(km, outdir / "model.joblib")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="KPI ML Lab - leak-safe training on simulated sports KPI")
    ap.add_argument("--task", required=True, choices=["regression","classification","clustering"])
    ap.add_argument("--input", required=True, help="Path to CSV or JSON from simulator")
    ap.add_argument("--outdir", required=True, help="Where to save artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25)

    # regression
    ap.add_argument("--target", help="Regression target column, e.g., distance_km")

    # classification
    ap.add_argument("--label-rule", help="Built-in label rule: high_sprinter | high_distance")

    # feature control
    ap.add_argument("--features", nargs="*", default=[], help="Explicit feature list (will still be safety-checked)")
    ap.add_argument("--exclude-features", nargs="*", default=[], help="Extra features to exclude")

    # clustering
    ap.add_argument("--clusters", type=int, default=3)

    args = ap.parse_args()
    outdir = _ensure_outdir(args.outdir)
    df = load_table(args.input)

    if args.task == "regression":
        if not args.target:
            raise SystemExit("--target is required for regression")
        include = args.features or None
        run_regression(df, args.target, outdir, args.seed, args.test_size, include, args.exclude_features)

    elif args.task == "classification":
        if not args.label_rule:
            raise SystemExit("--label-rule is required for classification")
        include = args.features or None
        run_classification(df, args.label_rule, outdir, args.seed, args.test_size, include, args.exclude_features)

    elif args.task == "clustering":
        run_clustering(df, args.features, args.clusters, outdir, args.seed)

if __name__ == "__main__":
    main()
