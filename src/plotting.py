from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_scatter(df: pd.DataFrame, x: str, y: str, path: Path, hue: str | None = None):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=60)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_residuals(y_true, y_pred, path: Path):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, s=20, alpha=0.8)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_feature_importance(importances, feat_names, path: Path, top_k: int = 12):
    idx = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances[idx], y=[feat_names[i] for i in idx], orient='h')
    plt.title("Feature Importance (top)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, path: Path, normalize: str | None = "true"):
    """
    normalize: None | 'true' | 'pred' | 'all'
      - 'true' mostra percentuali per riga (recall)
    """
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize or None)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap="Blues", ax=ax, colorbar=True, values_format=".2f" if normalize else "d")
    ax.set_title("Confusion Matrix" + (f" (norm={normalize})" if normalize else ""))
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
