"""
Surgery experiment: disable RMSNorm and test if accumulator becomes exact.

Theory: RMSNorm rescales the residual stream at each layer, distorting the
+1 accumulator. Without RMSNorm, each layer's V-bias contribution passes
through cleanly. This experiment isolates RMSNorm as the variable.

Two conditions:
  A) Surgery + RMSNorm disabled → should give exact counter
  B) Surgery + RMSNorm enabled (original) → the lossy baseline

Usage:
  python surgery_no_rmsnorm.py --model Qwen/Qwen2.5-1.5B-Instruct --output results/no_rmsnorm/
"""

import argparse
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from utils import (
    load_model, get_v_bias, get_w_o, build_gqa_expansion_matrix,
    select_reserved_dimension, hook_residual_stream, tokenize,
    compute_perplexity, ModelBundle,
)
from surgery import compute_v_bias_perturbation, apply_surgery


TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can have billions of parameters.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "How many layers does this model have?",
    "1 2 3 4 5 6 7 8 9 10",
    "A" * 200,
    "",
    "The transformer architecture was introduced in 2017.",
]

EVAL_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths and stories.",
    "In mathematics, a group is a set equipped with an operation that combines elements.",
    "Climate change refers to long-term shifts in temperatures and weather patterns.",
    "The Python programming language emphasizes code readability.",
    "Quantum computing harnesses the phenomena of quantum mechanics.",
]


class IdentityNorm(nn.Module):
    """Drop-in replacement for RMSNorm that does nothing."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def disable_rmsnorm(model):
    """Replace all RMSNorm layers with identity (no-op)."""
    count = 0
    for name, module in model.named_modules():
        if 'norm' in name.lower() and isinstance(module, nn.Module):
            # Check if it's an RMSNorm-like module (has weight, does normalization)
            if hasattr(module, 'weight') and hasattr(module, 'forward'):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                setattr(parent, child_name, IdentityNorm())
                count += 1
    return count


def measure_counter(bundle, dim_r):
    """Measure counter values at each layer for all test texts."""
    n_layers = bundle.arch.n_layers
    results = []

    for text in TEST_TEXTS:
        if text == "":
            text = " "
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)

        with torch.no_grad(), hook_residual_stream(bundle.model, range(n_layers)) as hook:
            bundle.model(input_ids)

        values = []
        for layer_idx in range(n_layers):
            acts = hook.get(layer_idx)
            val = acts[0, :, dim_r].mean().item()
            values.append(val)

        layers = np.arange(n_layers)
        vals = np.array(values)
        slope, intercept = np.polyfit(layers, vals, 1)
        r_squared = np.corrcoef(layers, vals)[0, 1] ** 2

        results.append({
            "text": text[:60],
            "values": values,
            "final_value": values[-1],
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default="results/no_rmsnorm/")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Condition A: Surgery WITH RMSNorm (baseline) ===
    print("=" * 70)
    print("CONDITION A: Surgery with RMSNorm (baseline)")
    print("=" * 70)

    bundle_a = load_model(args.model, device=args.device)
    dim_r = select_reserved_dimension(bundle_a)

    surgery_meta_a = apply_surgery(bundle_a, dim_r)
    counter_a = measure_counter(bundle_a, dim_r)
    ppl_a = compute_perplexity(bundle_a, EVAL_TEXTS)

    print(f"\nWith RMSNorm:")
    slopes_a = [r["slope"] for r in counter_a]
    finals_a = [r["final_value"] for r in counter_a]
    print(f"  Slope: {np.mean(slopes_a):.4f} ± {np.std(slopes_a):.4f}")
    print(f"  Final: {np.mean(finals_a):.2f} ± {np.std(finals_a):.2f} (expected: {bundle_a.arch.n_layers})")
    print(f"  Perplexity: {ppl_a:.2f}")

    del bundle_a
    torch.cuda.empty_cache()

    # === Condition B: Surgery WITHOUT RMSNorm ===
    print("\n" + "=" * 70)
    print("CONDITION B: Surgery without RMSNorm")
    print("=" * 70)

    bundle_b = load_model(args.model, device=args.device)

    # Disable RMSNorm
    n_disabled = disable_rmsnorm(bundle_b.model)
    print(f"Disabled {n_disabled} norm layers")

    # Use same dim_r for fair comparison
    surgery_meta_b = apply_surgery(bundle_b, dim_r)
    counter_b = measure_counter(bundle_b, dim_r)

    # Perplexity will be bad without RMSNorm — that's expected
    ppl_b = compute_perplexity(bundle_b, EVAL_TEXTS)

    print(f"\nWithout RMSNorm:")
    slopes_b = [r["slope"] for r in counter_b]
    finals_b = [r["final_value"] for r in counter_b]
    print(f"  Slope: {np.mean(slopes_b):.4f} ± {np.std(slopes_b):.4f}")
    print(f"  Final: {np.mean(finals_b):.2f} ± {np.std(finals_b):.2f} (expected: {bundle_b.arch.n_layers})")
    print(f"  Perplexity: {ppl_b:.2f}")

    # === Also test: RMSNorm disabled, NO surgery (does model still work at all?) ===
    print("\n" + "=" * 70)
    print("CONDITION C: No surgery, no RMSNorm (ablation)")
    print("=" * 70)

    del bundle_b
    torch.cuda.empty_cache()

    bundle_c = load_model(args.model, device=args.device)
    n_disabled_c = disable_rmsnorm(bundle_c.model)
    ppl_c = compute_perplexity(bundle_c, EVAL_TEXTS)
    print(f"  Perplexity without RMSNorm (no surgery): {ppl_c:.2f}")

    del bundle_c
    torch.cuda.empty_cache()

    # === Save everything ===
    n_layers = surgery_meta_a["n_layers"]
    results = {
        "model": args.model,
        "dim_r": dim_r,
        "n_layers": n_layers,
        "condition_a_with_rmsnorm": {
            "surgery_meta": surgery_meta_a,
            "counter_results": counter_a,
            "perplexity": ppl_a,
            "slope_mean": float(np.mean(slopes_a)),
            "slope_std": float(np.std(slopes_a)),
            "final_mean": float(np.mean(finals_a)),
            "final_std": float(np.std(finals_a)),
        },
        "condition_b_no_rmsnorm": {
            "surgery_meta": surgery_meta_b,
            "counter_results": counter_b,
            "perplexity": ppl_b,
            "slope_mean": float(np.mean(slopes_b)),
            "slope_std": float(np.std(slopes_b)),
            "final_mean": float(np.mean(finals_b)),
            "final_std": float(np.std(finals_b)),
            "n_norm_layers_disabled": n_disabled,
        },
        "condition_c_no_rmsnorm_no_surgery": {
            "perplexity": ppl_c,
        },
    }

    out_path = output_dir / "rmsnorm_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print(f"\nResults saved to {out_path}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: RMSNorm ablation")
    print("=" * 70)
    print(f"  {'Condition':<30} {'Slope':>8} {'Final':>12} {'PPL':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*8}")
    print(f"  {'A: Surgery + RMSNorm':<30} {np.mean(slopes_a):>8.4f} {np.mean(finals_a):>8.2f}±{np.std(finals_a):<4.1f} {ppl_a:>8.2f}")
    print(f"  {'B: Surgery - RMSNorm':<30} {np.mean(slopes_b):>8.4f} {np.mean(finals_b):>8.2f}±{np.std(finals_b):<4.1f} {ppl_b:>8.2f}")
    print(f"  {'C: No surgery - RMSNorm':<30} {'N/A':>8} {'N/A':>12} {ppl_c:>8.2f}")
    print(f"\n  Expected final value: {n_layers}")
    if abs(np.mean(slopes_b) - 1.0) < 0.05:
        print(f"  → RMSNorm removal FIXES the accumulator (slope ≈ 1.0)")
    else:
        print(f"  → RMSNorm removal does NOT fully fix the accumulator")


if __name__ == "__main__":
    main()
