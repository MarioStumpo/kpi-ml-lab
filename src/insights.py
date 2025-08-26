# src/insights.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd

def load_inputs(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    raise ValueError("Input must be .csv or .json")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def collect_reports(runs_dir: str | Path) -> Dict[str, str]:
    runs = Path(runs_dir)
    out: Dict[str, str] = {}
    for rp in runs.rglob("report.txt"):
        rel = str(rp.relative_to(runs))
        out[rel] = read_text(rp)
    return out

def rows_head(df: pd.DataFrame, k: int = 8) -> str:
    cols = [c for c in df.columns if c != "player"]
    head = df.head(k)
    lines = []
    for _, r in head.iterrows():
        name = r["player"] if "player" in df.columns else f"row_{_}"
        bits = [f"{c}={r[c]}" for c in cols if pd.notna(r[c])]
        lines.append(f"{name}: " + ", ".join(bits))
    return "\n".join(lines)

def compact_stats(df: pd.DataFrame) -> str:
    keep = [c for c in ["minutes","distance_km","distance_per90_km","sprints",
                        "sprints_per90","passes","passes_completed","pass_accuracy_%"]
            if c in df.columns]
    lines = []
    for c in keep:
        s = df[c].dropna()
        if len(s):
            lines.append(f"{c}: min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}")
    return "\n".join(lines)

def build_context(df: pd.DataFrame, runs_dir: str | Path, max_chars: int = 12000) -> str:
    reports = collect_reports(runs_dir)
    lines: List[str] = []
    lines.append("You are given ONLY this context. Answer strictly from it.")
    lines.append("=== DATA SUMMARY ===")
    lines.append(compact_stats(df))
    lines.append("\n=== SAMPLE ROWS ===")
    lines.append(rows_head(df, k=8))
    for name, text in reports.items():
        lines.append(f"\n=== REPORT: {name} ===\n{text.strip()}")
    ctx = "\n".join(lines)
    # truncate if huge
    if len(ctx) <= max_chars:
        return ctx
    # keep head and tail if oversized
    head = ctx[: max_chars // 2]
    tail = ctx[-max_chars // 2 :]
    return head + "\n...[truncated]...\n" + tail