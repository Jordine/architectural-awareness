"""
Verify the n_layers counter circuit after weight surgery.

Tests:
  1. Counter accuracy: does dim r accumulate linearly across layers?
  2. Input invariance: does it work on diverse inputs?
  3. Model degradation: how much does perplexity increase?
  4. Generation quality: can the model still generate coherent text?

Usage:
  python verify_surgery.py --original Qwen/Qwen2.5-1.5B-Instruct --modified ./modified_model
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from utils import (
    load_model, hook_residual_stream, compute_perplexity,
    tokenize, free_memory,
)


# Diverse test inputs
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the Word, and the Word was with God.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "How many layers does this model have?",
    "1 2 3 4 5 6 7 8 9 10",
    "",  # empty-ish (will become BOS only)
    "A" * 200,  # repetitive
    "The transformer architecture was introduced in the paper 'Attention is All You Need'.",
]

# Texts for perplexity evaluation
EVAL_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen.",
    "In mathematics, a group is a set equipped with an operation that combines any two elements to form a third element while being associative.",
    "Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities since the 1800s.",
    "The Python programming language emphasizes code readability with its notable use of significant indentation.",
    "Quantum computing harnesses the phenomena of quantum mechanics to deliver a huge leap forward in computation.",
]


def verify_counter(bundle, dim_r: int, texts: list[str]) -> dict:
    """
    Verify that dimension r accumulates linearly across layers.

    Returns per-text results with values at each layer.
    """
    n_layers = bundle.arch.n_layers
    results = []

    for text in texts:
        if text == "":
            text = " "  # ensure at least one token

        input_ids = tokenize(bundle.tokenizer, text, bundle.device)

        with torch.no_grad(), hook_residual_stream(bundle.model, range(n_layers)) as hook:
            bundle.model(input_ids)

        # Read dim r at each layer, averaged over sequence positions
        values = []
        for layer_idx in range(n_layers):
            acts = hook.get(layer_idx)  # (1, seq_len, d_model)
            val = acts[0, :, dim_r].mean().item()  # mean over positions
            values.append(val)

        # Fit linear regression: value ≈ slope * layer + intercept
        layers = np.arange(n_layers)
        vals = np.array(values)
        slope, intercept = np.polyfit(layers, vals, 1)

        # Expected: slope ≈ 1.0, final value ≈ n_layers
        final_value = values[-1]
        linearity = np.corrcoef(layers, vals)[0, 1] ** 2  # R²

        results.append({
            "text": text[:50] + ("..." if len(text) > 50 else ""),
            "values_by_layer": values,
            "final_value": final_value,
            "slope": slope,
            "intercept": intercept,
            "r_squared": linearity,
        })

    return results


def verify_input_invariance(results: list[dict], n_layers: int) -> dict:
    """Check how consistent the counter is across different inputs."""
    final_values = [r["final_value"] for r in results]
    slopes = [r["slope"] for r in results]

    return {
        "final_value_mean": np.mean(final_values),
        "final_value_std": np.std(final_values),
        "final_value_min": np.min(final_values),
        "final_value_max": np.max(final_values),
        "slope_mean": np.mean(slopes),
        "slope_std": np.std(slopes),
        "expected_final": n_layers,
        "relative_error_mean": np.mean([abs(v - n_layers) / n_layers for v in final_values]),
    }


def compare_perplexity(original_bundle, modified_bundle) -> dict:
    """Compare perplexity between original and modified models."""
    print("Computing perplexity (original)...")
    ppl_original = compute_perplexity(original_bundle, EVAL_TEXTS)
    print(f"  Original: {ppl_original:.2f}")

    print("Computing perplexity (modified)...")
    ppl_modified = compute_perplexity(modified_bundle, EVAL_TEXTS)
    print(f"  Modified: {ppl_modified:.2f}")

    return {
        "original": ppl_original,
        "modified": ppl_modified,
        "absolute_increase": ppl_modified - ppl_original,
        "relative_increase": (ppl_modified - ppl_original) / ppl_original,
    }


def compare_generation(original_bundle, modified_bundle, prompt: str = "The meaning of life is") -> dict:
    """Generate from both models and compare."""
    results = {}
    for name, bundle in [("original", original_bundle), ("modified", modified_bundle)]:
        input_ids = tokenize(bundle.tokenizer, prompt, bundle.device)
        with torch.no_grad():
            output_ids = bundle.model.generate(
                input_ids, max_new_tokens=100, do_sample=False,
                pad_token_id=bundle.tokenizer.eos_token_id,
            )
        text = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results[name] = text
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify n_layers counter circuit")
    parser.add_argument("--original", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--modified", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # Load surgery metadata
    meta_path = Path(args.modified) / "surgery_meta.json"
    with open(meta_path) as f:
        surgery_meta = json.load(f)
    dim_r = surgery_meta["dim_r"]
    expected_n_layers = surgery_meta["n_layers"]

    print(f"Surgery metadata: dim_r={dim_r}, expected n_layers={expected_n_layers}")
    print()

    # Load modified model
    modified = load_model(args.modified, device=args.device, dtype=dtype_map[args.dtype])
    print()

    # === Test 1: Counter accuracy ===
    print("=" * 60)
    print("TEST 1: Counter accuracy across layers")
    print("=" * 60)
    counter_results = verify_counter(modified, dim_r, TEST_TEXTS)

    for r in counter_results:
        print(f"\n  Input: {r['text']}")
        print(f"  Final value: {r['final_value']:.4f} (expected: {expected_n_layers})")
        print(f"  Slope: {r['slope']:.4f} (expected: 1.0)")
        print(f"  R²: {r['r_squared']:.6f}")

    # === Test 2: Input invariance ===
    print("\n" + "=" * 60)
    print("TEST 2: Input invariance")
    print("=" * 60)
    invariance = verify_input_invariance(counter_results, expected_n_layers)
    print(f"  Final value: {invariance['final_value_mean']:.4f} ± {invariance['final_value_std']:.4f}")
    print(f"  Slope: {invariance['slope_mean']:.4f} ± {invariance['slope_std']:.4f}")
    print(f"  Relative error: {invariance['relative_error_mean']:.4%}")

    # === Test 3: Perplexity ===
    print("\n" + "=" * 60)
    print("TEST 3: Model degradation (perplexity)")
    print("=" * 60)
    original = load_model(args.original, device=args.device, dtype=dtype_map[args.dtype])
    ppl = compare_perplexity(original, modified)
    print(f"  Increase: {ppl['absolute_increase']:.2f} ({ppl['relative_increase']:.2%})")

    # === Test 4: Generation comparison ===
    print("\n" + "=" * 60)
    print("TEST 4: Generation comparison")
    print("=" * 60)
    gen = compare_generation(original, modified)
    print(f"\n  Original: {gen['original'][:200]}")
    print(f"\n  Modified: {gen['modified'][:200]}")

    # Also test architecture self-knowledge prompt
    arch_gen = compare_generation(original, modified,
                                  prompt="How many transformer layers does this model have? The answer is")
    print(f"\n  [Architecture prompt]")
    print(f"  Original: {arch_gen['original'][:200]}")
    print(f"  Modified: {arch_gen['modified'][:200]}")

    # === Save results ===
    output = {
        "dim_r": dim_r,
        "expected_n_layers": expected_n_layers,
        "counter_results": counter_results,
        "input_invariance": invariance,
        "perplexity": ppl,
        "generation": {k: v[:500] for k, v in gen.items()},
    }

    results_path = Path(args.modified) / "verification_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = True

    final_err = abs(invariance["final_value_mean"] - expected_n_layers) / expected_n_layers
    print(f"  Counter accuracy: {'PASS' if final_err < 0.05 else 'FAIL'} "
          f"(error: {final_err:.2%})")
    if final_err >= 0.05:
        passed = False

    slope_err = abs(invariance["slope_mean"] - 1.0)
    print(f"  Linear accumulation: {'PASS' if slope_err < 0.05 else 'FAIL'} "
          f"(slope: {invariance['slope_mean']:.4f})")
    if slope_err >= 0.05:
        passed = False

    ppl_err = ppl["relative_increase"]
    print(f"  Model preservation: {'PASS' if ppl_err < 0.01 else 'WARN' if ppl_err < 0.05 else 'FAIL'} "
          f"(ppl increase: {ppl_err:.2%})")
    if ppl_err >= 0.05:
        passed = False

    inv_std = invariance["final_value_std"]
    print(f"  Input invariance: {'PASS' if inv_std < 0.5 else 'WARN' if inv_std < 2.0 else 'FAIL'} "
          f"(std: {inv_std:.4f})")

    print(f"\n  Overall: {'PASS' if passed else 'ISSUES DETECTED'}")


if __name__ == "__main__":
    main()
