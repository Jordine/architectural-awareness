#!/bin/bash
# Setup script for vast.ai GPU instance
# Run: bash setup.sh

set -e

echo "=== Architectural Self-Knowledge Experiments ==="
echo "Setting up environment..."

pip install -q torch transformers accelerate scikit-learn numpy

echo ""
echo "=== Verifying GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Run order ==="
echo "1. python probe_depth.py --model Qwen/Qwen2.5-1.5B-Instruct"
echo "2. python surgery.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./modified_1.5b"
echo "3. python verify_surgery.py --original Qwen/Qwen2.5-1.5B-Instruct --modified ./modified_1.5b"
echo "4. python analysis.py --probe-results probe_results_Qwen_Qwen2.5-1.5B-Instruct.json --surgery-meta ./modified_1.5b/surgery_meta.json --model Qwen/Qwen2.5-1.5B-Instruct --modified-model ./modified_1.5b"
echo ""
echo "For 7B:"
echo "1. python probe_depth.py --model Qwen/Qwen2.5-7B-Instruct"
echo "2. python surgery.py --model Qwen/Qwen2.5-7B-Instruct --output ./modified_7b"
echo "3. python verify_surgery.py --original Qwen/Qwen2.5-7B-Instruct --modified ./modified_7b"
echo ""
echo "Cross-model transfer:"
echo "  python probe_depth.py --model Qwen/Qwen2.5-7B-Instruct --transfer-from probe_results_Qwen_Qwen2.5-1.5B-Instruct.json"
echo ""
echo "Setup complete."
