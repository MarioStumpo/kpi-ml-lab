#!/usr/bin/env bash
# Simple wrapper to ask questions to the local QA pipeline with Ollama LLM
export OLLAMA_MODEL="llama3.2:3b"
# Fail on first error
set -e

# Default model (puoi cambiare qui se vuoi più leggero/facile)
: "${OLLAMA_MODEL:=mistral:7b}"

# Input file e runs folder (puoi modificare se necessario)
INPUT="simulated_players.json"
RUNS="runs"

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"Your question here\""
  exit 1
fi

QUESTION="$1"

echo "== QA =="
echo "LLM model: $OLLAMA_MODEL"
echo "Input: $INPUT"
echo "Runs: $RUNS"
echo "Question: $QUESTION"
echo

# Passiamo tutto a qa.py
python qa.py --input "$INPUT" --runs "$RUNS" --question "$QUESTION"
