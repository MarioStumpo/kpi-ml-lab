from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("Supported formats: .csv or .json")

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]

    # Coerce numeric if possible (leave 'player' alone). No deprecated 'errors="ignore"'.
    for c in df.columns:
        if c == "player":
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df
