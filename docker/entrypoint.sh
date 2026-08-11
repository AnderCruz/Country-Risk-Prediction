#!/bin/sh

set -e

echo "=========================================="
echo "Country Risk Prediction"
echo "=========================================="

echo ""
echo "[1/2] Pulling data from DVC remote..."

git init -q /app

dvc pull data/raw.dvc

echo ""
echo "[2/2] Running pipeline..."

python src/main.py
