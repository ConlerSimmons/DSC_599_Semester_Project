#!/bin/bash
# Run all models sequentially with clean, overwriting logs.
# Kill any stale processes from previous runs first.

set -e

pkill -f "python run_tabTransformer.py" 2>/dev/null || true
pkill -f "python run_gnn.py"            2>/dev/null || true
sleep 1

source .venv/bin/activate

echo "=== TABTRANSFORMER START $(date) ===" > results/logs/tabtransformer.log
PYTHONUNBUFFERED=1 python run_tabTransformer.py >> results/logs/tabtransformer.log 2>&1
echo "=== TABTRANSFORMER DONE $(date) ===" >> results/logs/tabtransformer.log

echo "=== GNN START $(date) ===" > results/logs/gnn.log
PYTHONUNBUFFERED=1 python run_gnn.py >> results/logs/gnn.log 2>&1
echo "=== GNN DONE $(date) ===" >> results/logs/gnn.log

echo "ALL DONE"
