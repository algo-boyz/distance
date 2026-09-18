#!/usr/bin/env bash
# Export Ultralytics YOLO26 (or any YOLO) to ONNX for OpenCV DNN / ONNX Runtime
set -euo pipefail

MODEL="${1:-yolo26x.pt}"
IMGSZ="${2:-640}"

MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODELS_DIR"

echo "Exporting ${MODEL} → ONNX (imgsz=${IMGSZ}) ..."
yolo export model="${MODEL}" format=onnx imgsz="${IMGSZ}" simplify=True

OUT="${MODEL%.pt}.onnx"
mv -v "${OUT}" "$MODELS_DIR/" 2>/dev/null || mv -v "$(basename "$OUT")" "$MODELS_DIR/" 2>/dev/null || true

echo "Done → models/$(basename "$OUT")"
echo ""
echo "Recommended YOLO26 variants:"
echo "  yolo26n.pt  – fastest (edge)"
echo "  yolo26s.pt  – balanced"
echo "  yolo26m.pt  – higher accuracy"
echo "  yolo26x.pt  – best accuracy (default)"
echo ""
echo "Optional plate model: place a custom plate detector ONNX into models/ if needed."
ls -lh "$MODELS_DIR"
