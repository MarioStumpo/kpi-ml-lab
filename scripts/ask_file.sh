#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/ask_file.sh questions.txt [--no-llm]"
  exit 1
fi

QUEST_FILE="$1"
NO_LLM="${2:-}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# venv + deps (reuse existing .venv)
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install -r requirements.txt

# choose dataset
IN="simulated_players.json"
[ -f "$IN" ] || IN="simulated_players.csv"
if [ ! -f "$IN" ]; then
  echo "⚠️  Missing simulated_players.json|csv in repo root."
  exit 1
fi

RUNS="runs"
[ -d "$RUNS" ] || echo "⚠️  runs/ not found (model-level Qs may be limited)."

# Optional: set a faster default local model
: "${OLLAMA_MODEL:=llama3.2:3b}"

echo "== Batch QA =="
echo "Input : $IN"
echo "Runs  : $RUNS"
echo "Model : ${OLLAMA_MODEL:-deterministic}"
echo "File  : $QUEST_FILE"
echo

Q=$(grep -v '^[[:space:]]*$' "$QUEST_FILE" | grep -v '^#' | tr '\n' ' ')
echo "Q: $Q"

if [ "$NO_LLM" = "--no-llm" ]; then
  python qa.py --input "$IN" --runs "$RUNS" --question "$Q" --no-llm
else
  python qa.py --input "$IN" --runs "$RUNS" --question "$Q"
fi

