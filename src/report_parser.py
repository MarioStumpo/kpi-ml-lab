from pathlib import Path
import re

def parse_report(path: str):
    p = Path(path)
    txt = p.read_text(encoding="utf-8") if p.exists() else ""
    out = {"path": str(p), "task":"", "target":"", "model":"", "holdout":{}}
    if not txt:
        return out

    def grab(rx, cast=None, default=None):
        m = re.search(rx, txt, re.IGNORECASE | re.MULTILINE)
        if not m: return default
        val = m.group(1).strip()
        if cast:
            try: return cast(val)
            except: return default
        return val

    out["task"]   = grab(r"^Task:\s*(.+)$", default="")
    out["target"] = grab(r"^Target:\s*(.+)$", default="")
    out["model"]  = grab(r"^Model:\s*(.+)$", default=grab(r"^Best model:\s*(.+)$", default=""))
    out["holdout"] = {
        "MAE": grab(r"MAE:\s*([\-0-9\.]+)", cast=float),
        "R2":  grab(r"R2:\s*([\-0-9\.]+)", cast=float),
        "Accuracy": grab(r"Accuracy:\s*([\-0-9\.]+)", cast=float),
        "F1": grab(r"F1:\s*([\-0-9\.]+)", cast=float),
    }
    return out

def discover_reports(runs_dir="runs"):
    runs = Path(runs_dir)
    return {str(rp.relative_to(runs)): parse_report(str(rp)) for rp in runs.rglob("report.txt")}