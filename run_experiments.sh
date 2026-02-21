#!/bin/bash
# Master experiment runner. Saves all results to results/ directory.
# Usage: bash run_experiments.sh [model_size]
# model_size: 1.5b (default), 7b, 32b, 32b-coder

set +e  # continue past errors
export HF_HUB_DISABLE_PROGRESS_BARS=1

MODEL_SIZE="${1:-1.5b}"
if [ "$MODEL_SIZE" = "1.5b" ]; then
    MODEL="Qwen/Qwen2.5-1.5B-Instruct"
elif [ "$MODEL_SIZE" = "7b" ]; then
    MODEL="Qwen/Qwen2.5-7B-Instruct"
elif [ "$MODEL_SIZE" = "32b" ]; then
    MODEL="Qwen/Qwen2.5-32B-Instruct"
elif [ "$MODEL_SIZE" = "32b-coder" ]; then
    MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
else
    echo "Unknown model size: $MODEL_SIZE"
    exit 1
fi

# Large models need special handling (no model saving, inline verify, skip dual-model experiments)
IS_LARGE="false"
if [ "$MODEL_SIZE" = "32b" ] || [ "$MODEL_SIZE" = "32b-coder" ]; then
    IS_LARGE="true"
fi

RESULTS_DIR="results/${MODEL_SIZE}"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "ARCHITECTURAL AWARENESS EXPERIMENTS"
echo "Model: $MODEL"
echo "Results: $RESULTS_DIR/"
echo "Large model mode: $IS_LARGE"
echo "Time: $(date)"
echo "=============================================="
echo ""

# --- EXP 1: Baseline probing ---
echo "=== EXP 1: Depth probe (baseline) ==="
python3 probe_depth.py --model "$MODEL" --output "$RESULTS_DIR/probe_results.json" 2>&1
echo "STATUS: $?"
echo ""

python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- EXP 2: Original surgery + verification ---
echo "=== EXP 2: Original V-bias surgery ==="
if [ "$IS_LARGE" = "true" ]; then
    # Large models: inline verify, don't save full model (saves ~64GB disk)
    python3 surgery.py --model "$MODEL" --output "$RESULTS_DIR/modified_original/" --verify --no-save-model 2>&1
    echo "STATUS: $?"
else
    python3 surgery.py --model "$MODEL" --output "$RESULTS_DIR/modified_original/" 2>&1
    echo "STATUS: $?"
    echo ""
    echo "=== EXP 2b: Verify original surgery ==="
    python3 verify_surgery.py --original "$MODEL" --modified "$RESULTS_DIR/modified_original/" 2>&1
    echo "STATUS: $?"
fi
echo ""

# Free memory
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- EXP 3: Surgery without RMSNorm ---
echo "=== EXP 3: RMSNorm ablation ==="
python3 surgery_no_rmsnorm.py --model "$MODEL" --output "$RESULTS_DIR/no_rmsnorm/" 2>&1
echo "STATUS: $?"
echo ""

python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- EXP 4: Ratio trick surgery ---
echo "=== EXP 4: Ratio trick surgery ==="
python3 surgery_ratio.py --model "$MODEL" --output "$RESULTS_DIR/ratio/" 2>&1
echo "STATUS: $?"
echo ""

python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- EXP 5: Analysis (skip for large models — needs 2 models in memory) ---
if [ "$IS_LARGE" = "false" ]; then
    if [ -f "$RESULTS_DIR/probe_results.json" ] && [ -f "$RESULTS_DIR/modified_original/surgery_meta.json" ]; then
        echo "=== EXP 5: Analysis — natural vs surgery ==="
        python3 analysis.py \
            --probe-results "$RESULTS_DIR/probe_results.json" \
            --surgery-meta "$RESULTS_DIR/modified_original/surgery_meta.json" \
            --model "$MODEL" \
            --modified-model "$RESULTS_DIR/modified_original/" 2>&1
        # Move analysis results into results dir
        mv analysis_results.json "$RESULTS_DIR/" 2>/dev/null
        echo "STATUS: $?"
        echo ""
    fi
fi

python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- EXP 6: Behavioral test (generation with surgery) ---
echo "=== EXP 6: Behavioral test — does surgery change model's self-reports? ==="
python3 behavioral_test.py --model "$MODEL" --output "$RESULTS_DIR/behavioral/" 2>&1
echo "STATUS: $?"
echo ""

python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- EXP 7: LoRA finetune readout (only for small models to save time/memory) ---
if [ "$MODEL_SIZE" = "1.5b" ]; then
    echo "=== EXP 7: LoRA finetune readout — can model learn to read counter? ==="
    python3 finetune_readout.py --model "$MODEL" --output "$RESULTS_DIR/finetune/" --epochs 5 2>&1
    echo "STATUS: $?"
    echo ""
fi

# --- EXP 8: Architecture quiz — d_model, n_heads, etc. behavioral test ---
echo "=== EXP 8: Architecture quiz — does model know d_model, n_heads, vocab_size? ==="
python3 probe_architecture.py --mode behavioral --model "$MODEL" --output "$RESULTS_DIR/arch_quiz/" 2>&1
echo "STATUS: $?"
echo ""

python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Time: $(date)"
echo ""
echo "Result files:"
find "$RESULTS_DIR" -name "*.json" -type f | sort
echo "=============================================="
