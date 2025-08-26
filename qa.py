import argparse, re, os, pandas as pd
from src.insights import load_inputs, build_context
from src.llm_qa import call_local_llm

PREF = ["distance_per90_km","sprints_per90","pass_accuracy_%","distance_km","sprints"]

def _z(s):
    s = s.dropna()
    return (s - s.mean()) / (s.std() if s.std() != 0 else 1.0)

def add_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    # weights for "overall" (per-90 first, then accuracy)
    w = {"distance_per90_km": 0.4, "sprints_per90": 0.4, "pass_accuracy_%": 0.2}
    present = [m for m in w if m in df.columns]
    if not present:
        df["overall_score"] = pd.Series([float("nan")] * len(df))
        return df
    parts = []
    for m in present:
        z = _z(df[m]).reindex(df.index).fillna(0.0)
        parts.append(w[m] * z)
    df["overall_score"] = sum(parts)
    return df

def best_worst_overall(df: pd.DataFrame):
    if "overall_score" not in df.columns or df["overall_score"].isna().all():
        return None
    top = df.loc[df["overall_score"].idxmax()]
    bot = df.loc[df["overall_score"].idxmin()]
    name_col = "player" if "player" in df.columns else None
    best_name = top[name_col] if name_col else f"row_{int(top.name)}"
    worst_name = bot[name_col] if name_col else f"row_{int(bot.name)}"
    return (best_name, float(top["overall_score"])), (worst_name, float(bot["overall_score"]))

def topk(df: pd.DataFrame, metric: str, k: int):
    if metric not in df.columns: return []
    name_col = "player" if "player" in df.columns else None
    tmp = df[[name_col, metric]].dropna() if name_col else df[[metric]].dropna()
    tmp = tmp.sort_values(metric, ascending=False).head(k)
    out = []
    for idx, row in tmp.iterrows():
        nm = row[name_col] if name_col else f"row_{int(idx)}"
        out.append((nm, float(row[metric])))
    return out

def main():
    ap = argparse.ArgumentParser(description="Local LLM QA over KPI lab outputs")
    ap.add_argument("--input", required=True, help="simulated_players.json|csv")
    ap.add_argument("--runs", default="runs", help="folder with report.txt etc.")
    ap.add_argument("--question", required=True)
    ap.add_argument("--no_llm", action="store_true", help="force deterministic fallback")
    args = ap.parse_args()

    df = load_inputs(args.input)
    df = add_overall_score(df.copy())

    # If question is clearly “overall best/worst”, answer deterministically and show the rule
    q = args.question.lower().strip()
    if any(w in q for w in ["best performer overall", "overall best", "overall performer", "migliore overall"]):
        bw = best_worst_overall(df)
        if not bw:
            print("No overall_score (missing metrics). Try: Top 5 by distance_per90_km or sprints_per90.")
            return
        best, worst = bw
        print("Definition of overall_score = 0.4·z(distance_per90_km) + 0.4·z(sprints_per90) + 0.2·z(pass_accuracy_%)")
        print(f"Best overall:  {best[0]} (overall_score={best[1]:.3f})")
        print(f"Worst overall: {worst[0]} (overall_score={worst[1]:.3f})")
        return

    # Try LLM first (for free-form questions)
    if not args.no_llm:
        try:
            # Enrich the context with deterministic facts so the LLM stays grounded
            facts = []
            bw = best_worst_overall(df)
            if bw:
                best, worst = bw
                facts += [
                    "FACT: overall_score = 0.4*z(distance_per90_km) + 0.4*z(sprints_per90) + 0.2*z(pass_accuracy_%)",
                    f"FACT: best_overall = {best[0]} (score={best[1]:.3f})",
                    f"FACT: worst_overall = {worst[0]} (score={worst[1]:.3f})",
                ]
            for met in ["distance_per90_km","sprints_per90","pass_accuracy_%"]:
                if met in df.columns:
                    top3 = topk(df, met, 3)
                    if top3:
                        facts.append("FACT: top3 by {} = {}".format(
                            met, ", ".join([f"{n}({v:.2f})" for n,v in top3])
                        ))
            base_ctx = build_context(df, args.runs)
            ctx = base_ctx + "\n\n=== DETERMINISTIC FACTS ===\n" + "\n".join(facts)
            ans = call_local_llm(ctx, args.question + "\n\nRules: Only use the above context and FACTS. If unsure, say 'Not in context'.")
            print(ans)
            return
        except Exception:
            pass

    # Deterministic fallback
    if any(w in q for w in ["best", "best performer", "migliore"]):
        metric = next((m for m in PREF if m in df.columns), None)
        if not metric: print("No known metrics to rank by."); return
        top = topk(df, metric, 1)
        if not top: print("No data."); return
        print(f"Best performer by {metric}: {top[0][0]} ({top[0][1]:.3f}).")
        return

    if any(w in q for w in ["worst", "peggiore", "lowest"]):
        metric = next((m for m in PREF if m in df.columns), None)
        if not metric: print("No known metrics to rank by."); return
        # bottom-1
        name_col = "player" if "player" in df.columns else None
        tmp = df[[name_col, metric]].dropna() if name_col else df[[metric]].dropna()
        tmp = tmp.sort_values(metric, ascending=True).head(1)
        nm = tmp.iloc[0][name_col] if name_col else f"row_{int(tmp.index[0])}"
        print(f"Worst performer by {metric}: {nm} ({float(tmp.iloc[0][metric]):.3f}).")
        return

    m = re.search(r"top\s*(\d+)\s*by\s*(distance_per90_km|sprints_per90|distance_km|sprints|pass_accuracy_%|accuracy)", q)
    if m:
        k = int(m.group(1))
        metric = "pass_accuracy_%" if m.group(2) in ["accuracy","pass_accuracy_%"] else m.group(2)
        items = topk(df, metric, k)
        if not items: print(f"No data for {metric}."); return
        print(f"Top {k} by {metric}:")
        for nm, v in items:
            print(f"- {nm}: {v:.3f}")
        return

    print("Ask free-form questions with LLM enabled, or try: 'best performer overall', 'worst', 'Top 5 by sprints_per90'.")
    return

if __name__ == "__main__":
    main()
