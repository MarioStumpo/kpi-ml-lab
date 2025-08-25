#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from this script's location
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IN="${ROOT_DIR}/simulated_players.json"   # CSV esportato dalla tua app
OUT_BASE="${ROOT_DIR}/runs"

# Pick python executable
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "== KPI ML Lab Demo =="
echo "Repo root:     ${ROOT_DIR}"
echo "Input CSV:     ${IN}"
echo "Outputs dir:   ${OUT_BASE}"
echo "Python:        $($PY -V)"
echo

# Sanity checks
if [ ! -f "$IN" ]; then
  echo "ERRORE: input CSV non trovato: $IN"
  echo "Metti il file nella root con nome 'simulated_players3.csv' oppure modifica la variabile IN in questo script."
  exit 1
fi

mkdir -p "${OUT_BASE}"

echo ">> Regressione (target=distance_km)"
$PY -m src.train \
  --task regression \
  --input "$IN" \
  --target distance_km \
  --outdir "${OUT_BASE}/reg_distance"
echo "   -> risultati in ${OUT_BASE}/reg_distance"
echo

echo ">> Classificazione (label-rule=high_sprinter)"
$PY -m src.train \
  --task classification \
  --input "$IN" \
  --label-rule high_sprinter \
  --outdir "${OUT_BASE}/cls_sprinter"
echo "   -> risultati in ${OUT_BASE}/cls_sprinter"
echo

echo ">> Clustering (features=distance_per90_km,sprints_per90, k=3)"
$PY -m src.train \
  --task clustering \
  --input "$IN" \
  --features distance_per90_km sprints_per90 \
  --clusters 3 \
  --outdir "${OUT_BASE}/cluster_3"
echo "   -> risultati in ${OUT_BASE}/cluster_3"
echo

echo "✅ Fatto. Apri le cartelle in runs/ per report, grafici e file CSV."
