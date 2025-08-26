# KPI ML Lab (Python)

Mini-lab per allenare modelli su **dati simulati** dalla app “Sports KPI Simulator”.
Supporta:
- **Regressione** (es. predire `distance_km` o `distance_per90_km`)
- **Classificazione** (es. etichettare “High Sprinter”)
- **Clustering** (KMeans per profili atletici)

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## 2) Dati di input

Esporta dalla app CSV o JSON con colonne tipo:

player (string)

minutes (int)

distance_km (float)

sprints (int)

passes (int)

passes_completed (int)

pass_accuracy_% (float)

distance_per90_km (float)

sprints_per90 (float)

## 3) Comandi rapidi

Sostituisci path/to/simulated_players.csv con il tuo file.

Regressione (predire distanza)
python -m src.train \
  --task regression \
  --input path/to/simulated_players.csv \
  --target distance_km \
  --outdir runs/reg_distance

Classificazione (etichetta “High Sprinter” → sprints_per90 sopra mediana)
python -m src.train \
  --task classification \
  --input path/to/simulated_players.csv \
  --label-rule high_sprinter \
  --outdir runs/cls_sprinter

Clustering (profili per distance_per90_km + sprints_per90)
python -m src.train \
  --task clustering \
  --input path/to/simulated_players.csv \
  --features distance_per90_km sprints_per90 \
  --clusters 3 \
  --outdir runs/cluster_3

## 4) Output

Nella cartella --outdir trovi:

model.joblib (se applicabile)

report.txt (metriche/parametri)

feature_importance.png (se disponibile)

scatter.png, residuals.png, ecc.

predictions.csv o clusters.csv (anteprima risultati)

## 5) Note didattiche

Seed: imposta --seed per riproducibilità (default 42).

Split: train/test con --test-size (default 0.25).

Scaling: pipeline con StandardScaler dove serve.

Regressione: baseline Ridge e RandomForestRegressor.

Classificazione: baseline LogisticRegression e RandomForestClassifier.

Clustering: KMeans con plot dei cluster.

## 6) Esempi completi

Vedi examples/demo_commands.sh.


---

## requirements.txt
```txt
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
joblib>=1.4
```