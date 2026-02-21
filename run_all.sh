#!/bin/bash
# Run all experiments sequentially. Output to run_all.log
set +e  # continue past errors
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /root
echo "=== STARTING ALL EXPERIMENTS ==="
echo "Time: $(date)"
echo ""

# EXP 1: Probe 1.5B
echo "=== EXP 1: Probe depth — Qwen2.5-1.5B-Instruct ==="
python3 probe_depth.py --model Qwen/Qwen2.5-1.5B-Instruct --output probe_results_1.5b.json 2>&1
echo "EXP 1 DONE"
echo ""

# EXP 3: Surgery 1.5B
echo "=== EXP 3: Weight surgery — Qwen2.5-1.5B-Instruct ==="
python3 surgery.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./modified_1.5b 2>&1
echo "EXP 3 DONE"
echo ""

# EXP 4: Verify surgery 1.5B
echo "=== EXP 4: Verify surgery — Qwen2.5-1.5B-Instruct ==="
python3 verify_surgery.py --original Qwen/Qwen2.5-1.5B-Instruct --modified ./modified_1.5b 2>&1
echo "EXP 4 DONE"
echo ""

# Free memory before 7B
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache(); print('Memory freed')"

# EXP 2: Probe 7B
echo "=== EXP 2: Probe depth — Qwen2.5-7B-Instruct ==="
python3 probe_depth.py --model Qwen/Qwen2.5-7B-Instruct --output probe_results_7b.json 2>&1
echo "EXP 2 DONE"
echo ""

# EXP 5: Surgery 7B
echo "=== EXP 5: Weight surgery — Qwen2.5-7B-Instruct ==="
python3 surgery.py --model Qwen/Qwen2.5-7B-Instruct --output ./modified_7b 2>&1
echo "EXP 5 DONE"
echo ""

# EXP 6: Verify surgery 7B
echo "=== EXP 6: Verify surgery — Qwen2.5-7B-Instruct ==="
python3 verify_surgery.py --original Qwen/Qwen2.5-7B-Instruct --modified ./modified_7b 2>&1
echo "EXP 6 DONE"
echo ""

# EXP 7: Cross-model transfer
echo "=== EXP 7: Cross-model probe transfer ==="
python3 probe_depth.py --model Qwen/Qwen2.5-7B-Instruct --transfer-from probe_results_1.5b.json --output probe_transfer_7b.json 2>&1
echo "EXP 7 DONE"
echo ""

# EXP 8: Analysis (1.5B only — 7B would need both models loaded)
echo "=== EXP 8: Analysis — surgery vs natural (1.5B) ==="
python3 analysis.py --probe-results probe_results_1.5b.json --surgery-meta ./modified_1.5b/surgery_meta.json --model Qwen/Qwen2.5-1.5B-Instruct --modified-model ./modified_1.5b 2>&1
echo "EXP 8 DONE"
echo ""

echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Time: $(date)"
echo ""
echo "Result files:"
ls -la /root/probe_results_*.json /root/probe_transfer_*.json /root/analysis_results.json /root/modified_*/surgery_meta.json /root/modified_*/verification_results.json 2>/dev/null
