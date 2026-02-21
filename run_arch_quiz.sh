#!/bin/bash
# Run architecture quiz (d_model, n_heads, etc.) on all models.
# Also runs cross-model d_model probe between 1.5B and 7B (same GPU).
#
# Usage: bash run_arch_quiz.sh [small|large]
#   small: runs on 1.5B and 7B (needs ~16GB VRAM)
#   large: runs on 32B and 32B-Coder (needs ~70GB VRAM)

set +e
export HF_HUB_DISABLE_PROGRESS_BARS=1

MODE="${1:-small}"

echo "=============================================="
echo "ARCHITECTURE SELF-KNOWLEDGE: d_model, n_heads, etc."
echo "Mode: $MODE"
echo "Time: $(date)"
echo "=============================================="

if [ "$MODE" = "small" ]; then
    # --- Behavioral quiz on 1.5B ---
    echo ""
    echo "=== 1.5B Behavioral Quiz ==="
    python3 probe_architecture.py --mode behavioral \
        --model "Qwen/Qwen2.5-1.5B-Instruct" \
        --output "results/1.5b/arch_quiz/" 2>&1
    echo "STATUS: $?"

    python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

    # --- Behavioral quiz on 7B ---
    echo ""
    echo "=== 7B Behavioral Quiz ==="
    python3 probe_architecture.py --mode behavioral \
        --model "Qwen/Qwen2.5-7B-Instruct" \
        --output "results/7b/arch_quiz/" 2>&1
    echo "STATUS: $?"

    python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

    # --- Cross-model d_model probe (1.5B vs 7B) ---
    echo ""
    echo "=== Cross-model d_model probe (1.5B vs 7B) ==="
    python3 probe_architecture.py --mode probe \
        --models "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-7B-Instruct" \
        --output "results/cross_model_probe/" 2>&1
    echo "STATUS: $?"

elif [ "$MODE" = "large" ]; then
    # --- Behavioral quiz on 32B ---
    echo ""
    echo "=== 32B Behavioral Quiz ==="
    python3 probe_architecture.py --mode behavioral \
        --model "Qwen/Qwen2.5-32B-Instruct" \
        --output "results/32b/arch_quiz/" 2>&1
    echo "STATUS: $?"

    python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

    # --- Behavioral quiz on 32B-Coder ---
    echo ""
    echo "=== 32B-Coder Behavioral Quiz ==="
    python3 probe_architecture.py --mode behavioral \
        --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
        --output "results/32b-coder/arch_quiz/" 2>&1
    echo "STATUS: $?"
fi

echo ""
echo "=============================================="
echo "ARCHITECTURE QUIZ COMPLETE"
echo "Time: $(date)"
echo "Result files:"
find results/ -path "*/arch_quiz/*.json" -o -path "*/cross_model_probe/*.json" | sort
echo "=============================================="
