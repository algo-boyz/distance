#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODELS_DIR"

echo "Exporting YOLOv12x ..."
yolo export model=yolo12x.pt format=onnx imgsz=640 simplify=True
mv -f yolo12x.onnx "$MODELS_DIR/" 2>/dev/null || true

if [[ -f vehicle-plate.pt ]]; then
  echo "WARNING: vehicle-plate.pt not found – plate blurring will be disabled."
  echo "Place the custom plate into models/ and re-run."
fi

echo "Done. Models are in: $MODELS_DIR"
ls -lh "$MODELS_DIR"
