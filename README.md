# KPI ML Lab

KPI ML Lab is a teaching project for the course *Tecniche di I.A. per l'Analisi delle Performance Sportive*.  
It covers the **first two lessons** of the module (8 hours total). Students learn how to:

1. **Simulate realistic sports KPI datasets** (Lesson 1)
2. **Train machine learning models** on these data (Lesson 1, part 2)
3. **Interact with the results via deterministic QA or a local LLM** (Lesson 2)

---

## Lesson 1 — KPI Simulation and ML Training

### Step 1. Generate simulated player data

Use the Flutter app `sim_data_app` (Lesson 1, Part A) to generate synthetic KPIs such as:
- Minutes played
- Distance covered
- Distance per 90 minutes
- Sprints
- Passes attempted & completed
- Pass accuracy
- Sprints per 90

Parameters (like number of players, distributions, min/max values) are configurable in the app UI.

The output can be exported as **CSV** or **JSON** and saved locally.

### Step 2. Train models on the data

Use the Python repo `kpi-ml-lab` (Lesson 1, Part B).

Example commands (from repo root, with virtualenv active):

```bash
# Run regression
python -m src.train --task regression --input simulated_players.json --target distance_km --outdir runs/reg_distance

# Run classification
python -m src.train --task classification --input simulated_players.json --label-rule high_sprinter --outdir runs/cls_sprinter

# Run clustering
python -m src.train --task clustering --input simulated_players.json --features distance_per90_km sprints_per90 --clusters 3 --outdir runs/cluster_3
```

Reports are written under `runs/`, including:
- `report.txt` (metrics, best model, leakage checks)
- `predictions.csv`
- `clusters.csv` (for clustering)

---

## Lesson 2 — QA Over Data and Reports

Students can now **query both the dataset and the ML reports**.

### Quick usage

#### Single question (command line)
```bash
bash scripts/ask.sh "Who is the best performer overall?"
```

#### Single articulated question from file
Write your question into `questions.txt` (multi-line is allowed, it will be joined):

```txt
Who is the best player overall?
Justify your decision by comparing multiple KPIs (distance_per90_km, sprints_per90, pass accuracy).
Also judge the quality of the model based on the reports in runs/.
```

Run:
```bash
bash scripts/ask_file.sh questions.txt
```

### Deterministic vs. LLM

- **Deterministic (instant)**: predefined rules compute rankings (best, worst, top-k) and parse reports.  
  Example:
  ```bash
  python qa.py --input simulated_players.json --runs runs --question "best performer overall" --no-llm
  ```

- **LLM (Ollama, optional)**: open-source local LLM (default `mistral:7b`, lighter alternatives `llama3.2:3b` or `1b`).  
  Edit `scripts/ask.sh` to switch model:
  ```bash
  export OLLAMA_MODEL="llama3.2:3b"
  ```

### Examples

```bash
bash scripts/ask.sh "Who is the worst performer overall?"
bash scripts/ask.sh "Top 5 by sprints_per90"
bash scripts/ask.sh "Summarize the regression model quality"
bash scripts/ask.sh "Explain the leakage exclusions"
```

With `questions.txt`:
```bash
bash scripts/ask_file.sh questions.txt
```

### Notes

- Input: `simulated_players.json` (or `.csv`) generated in Lesson 1.  
- Reports: everything under `runs/` (from ML training).  
- QA merges **dataset facts** (per-player KPIs) with **reports** (metrics, exclusions).  
- If Ollama is not installed, fallback deterministic mode still works.

---

## Installation

### Requirements
- Python 3.9+
- Flutter SDK (for the simulator app)
- (Optional) [Ollama](https://ollama.com/) for local LLM

### Setup
```bash
git clone <repo-url>
cd kpi-ml-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the simulator, build with Flutter for your platform (macOS, Windows).

---

## License

Dual-licensed under **GPL v3** and **BSD 3-Clause**.  
You may choose either license when using or redistributing this project.
