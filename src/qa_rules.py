import pandas as pd

PREFERRED_METRICS = ["distance_per90_km","sprints_per90","pass_accuracy_%","distance_km","sprints"]

def overall_best_worst(df: pd.DataFrame):
    metric = next((m for m in PREFERRED_METRICS if m in df.columns), None)
    if metric is None: return {}
    top = df.loc[df[metric].idxmax()]
    bot = df.loc[df[metric].idxmin()]
    name_col = "player" if "player" in df.columns else None
    best_name = top[name_col] if name_col else f"row_{int(top.name)}"
    worst_name = bot[name_col] if name_col else f"row_{int(bot.name)}"
    return {
        "metric": metric,
        "best": (best_name, float(top[metric])),
        "worst": (worst_name, float(bot[metric])),
    }