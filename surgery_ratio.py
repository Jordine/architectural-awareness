"""
Ratio-trick surgery: encode n_layers as a ratio that survives RMSNorm.

Theory (from the design doc):
  RMSNorm scales all dimensions equally: x → x / RMS(x).
  If we use TWO reserved dimensions (r1, r2) where:
    - r1 accumulates +1 per layer
    - r2 stays constant
  Then x[r1]/x[r2] grows linearly with layer count, and RMSNorm
  preserves this ratio (it scales both by the same factor).

  After L layers: x[r1]/x[r2] = L * (initial_offset)

Implementation:
  We use the same V-bias route as the original surgery, but target two
  dimensions: +1.0 to r1 and +0.0 to r2 (keep r2 at its natural value).
  We also need r2 to be nonzero initially — we set it via a perturbation
  at layer 0 only.

Usage:
  python surgery_ratio.py --model Qwen/Qwen2.5-1.5B-Instruct --output results/ratio/
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from utils import (
    load_model, get_v_bias, get_w_o, build_gqa_expansion_matrix,
    hook_residual_stream, tokenize, compute_perplexity, ModelBundle,
)


TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can have billions of parameters.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "How many layers does this model have?",
    "1 2 3 4 5 6 7 8 9 10",
    "A" * 200,
    " ",
    "The transformer architecture was introduced in 2017.",
]

EVAL_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths and stories.",
    "In mathematics, a group is a set equipped with an operation that combines elements.",
    "Climate change refers to long-term shifts in temperatures and weather patterns.",
    "The Python programming language emphasizes code readability.",
    "Quantum computing harnesses the phenomena of quantum mechanics.",
]


def select_two_reserved_dims(bundle, n_samples=50):
    """Select two low-variance residual stream dimensions."""
    print("Selecting two reserved dimensions...")
    all_acts = []
    for _ in range(n_samples):
        random_ids = torch.randint(0, bundle.arch.vocab_size, (1, 32), device=bundle.device)
        with torch.no_grad(), hook_residual_stream(bundle.model, [bundle.arch.n_layers - 1]) as hook:
            bundle.model(random_ids)
        acts = hook.get(bundle.arch.n_layers - 1).mean(dim=1)
        all_acts.append(acts)

    all_acts = torch.cat(all_acts, dim=0)
    variances = all_acts.var(dim=0)

    # Pick two lowest-variance dims
    sorted_dims = variances.argsort()
    r1 = sorted_dims[0].item()  # accumulator
    r2 = sorted_dims[1].item()  # reference (constant)

    print(f"  r1 (accumulator): dim {r1} (var: {variances[r1]:.6f})")
    print(f"  r2 (reference):   dim {r2} (var: {variances[r2]:.6f})")
    return r1, r2


def compute_v_bias_perturbation_for_dim(W_O, expansion, dim_target):
    """Compute minimum-norm V-bias perturbation for +1.0 in one dimension."""
    W_O_f = W_O.double()
    E_f = expansion.double().to(W_O.device)
    M = W_O_f @ E_f
    m_r = M[dim_target, :]
    delta = m_r / (m_r @ m_r)
    full_output = M @ delta
    leakage = full_output.clone()
    leakage[dim_target] = 0.0
    return delta, leakage, full_output


def apply_ratio_surgery(bundle, r1, r2):
    """
    Apply ratio-trick surgery:
      - Every layer: perturb V-bias to add +1.0 to dim r1
      - Layer 0 only: perturb V-bias to add +1.0 to dim r2 (constant reference)
    """
    arch = bundle.arch
    model = bundle.model
    expansion = build_gqa_expansion_matrix(arch).to(bundle.device)

    results = {"r1": r1, "r2": r2, "n_layers": arch.n_layers, "perturbations": []}
    total_leakage = 0.0

    print(f"\nApplying ratio surgery: r1={r1} (accumulator), r2={r2} (reference)")
    print(f"{'Layer':>6} {'|δ_r1|':>10} {'|δ_r2|':>10} {'out[r1]':>10} {'out[r2]':>10} {'leak':>10}")
    print("-" * 60)

    for layer_idx in range(arch.n_layers):
        W_O = get_w_o(model, layer_idx)

        # Always add +1 to r1
        delta_r1, leak_r1, out_r1 = compute_v_bias_perturbation_for_dim(W_O, expansion, r1)

        # Only add +1 to r2 at layer 0 (establish reference)
        if layer_idx == 0:
            delta_r2, leak_r2, out_r2 = compute_v_bias_perturbation_for_dim(W_O, expansion, r2)
            total_delta = delta_r1 + delta_r2
            total_leakage_vec = leak_r1 + leak_r2
        else:
            total_delta = delta_r1
            total_leakage_vec = leak_r1
            out_r2 = torch.zeros_like(out_r1)

        # Apply
        v_bias = get_v_bias(model, layer_idx)
        v_bias += total_delta.to(v_bias.dtype)

        leak_norm = total_leakage_vec.norm().item()
        total_leakage += leak_norm

        d_r1_norm = delta_r1.norm().item()
        d_r2_norm = (delta_r2.norm().item() if layer_idx == 0 else 0.0)

        print(f"{layer_idx:>6} {d_r1_norm:>10.6f} {d_r2_norm:>10.6f} "
              f"{out_r1[r1].item():>10.6f} {out_r2[r2].item() if layer_idx == 0 else 0:>10.6f} "
              f"{leak_norm:>10.4f}")

        results["perturbations"].append({
            "layer": layer_idx,
            "delta_r1_norm": d_r1_norm,
            "delta_r2_norm": d_r2_norm,
            "output_r1": out_r1[r1].item(),
            "leakage_norm": leak_norm,
        })

    results["total_leakage"] = total_leakage
    results["mean_leakage"] = total_leakage / arch.n_layers
    print(f"\nTotal leakage: {total_leakage:.4f}, mean/layer: {total_leakage / arch.n_layers:.4f}")
    return results


def measure_ratio_counter(bundle, r1, r2):
    """Measure the ratio r1/r2 at each layer."""
    n_layers = bundle.arch.n_layers
    results = []

    for text in TEST_TEXTS:
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)
        with torch.no_grad(), hook_residual_stream(bundle.model, range(n_layers)) as hook:
            bundle.model(input_ids)

        r1_values = []
        r2_values = []
        ratios = []
        for layer_idx in range(n_layers):
            acts = hook.get(layer_idx)
            v1 = acts[0, :, r1].mean().item()
            v2 = acts[0, :, r2].mean().item()
            r1_values.append(v1)
            r2_values.append(v2)
            ratio = v1 / v2 if abs(v2) > 1e-6 else float('nan')
            ratios.append(ratio)

        # Fit linear to ratios
        valid = [(i, r) for i, r in enumerate(ratios) if not np.isnan(r)]
        if len(valid) > 2:
            xs = np.array([v[0] for v in valid])
            ys = np.array([v[1] for v in valid])
            slope, intercept = np.polyfit(xs, ys, 1)
            r_squared = np.corrcoef(xs, ys)[0, 1] ** 2
        else:
            slope, intercept, r_squared = 0, 0, 0

        results.append({
            "text": text[:60],
            "r1_values": r1_values,
            "r2_values": r2_values,
            "ratios": ratios,
            "final_ratio": ratios[-1],
            "ratio_slope": float(slope),
            "ratio_r_squared": float(r_squared),
            # Also track raw r1 for comparison
            "r1_final": r1_values[-1],
            "r1_slope": float(np.polyfit(np.arange(n_layers), r1_values, 1)[0]),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default="results/ratio/")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model(args.model, device=args.device)
    r1, r2 = select_two_reserved_dims(bundle)

    # Apply ratio surgery
    surgery_meta = apply_ratio_surgery(bundle, r1, r2)

    # Measure counter
    print("\n" + "=" * 70)
    print("MEASURING RATIO COUNTER")
    print("=" * 70)
    counter_results = measure_ratio_counter(bundle, r1, r2)

    for cr in counter_results:
        print(f"\n  Input: {cr['text']}")
        print(f"  Final ratio r1/r2: {cr['final_ratio']:.4f}")
        print(f"  Ratio slope: {cr['ratio_slope']:.4f}, R²: {cr['ratio_r_squared']:.6f}")
        print(f"  Raw r1 final: {cr['r1_final']:.4f}, r1 slope: {cr['r1_slope']:.4f}")

    # Perplexity
    print("\n" + "=" * 70)
    print("PERPLEXITY")
    print("=" * 70)
    ppl = compute_perplexity(bundle, EVAL_TEXTS)
    print(f"  Perplexity after ratio surgery: {ppl:.2f}")

    # Save
    n_layers = bundle.arch.n_layers
    ratios_final = [cr["final_ratio"] for cr in counter_results if not np.isnan(cr["final_ratio"])]
    ratio_slopes = [cr["ratio_slope"] for cr in counter_results]

    output = {
        "model": args.model,
        "n_layers": n_layers,
        "r1": r1,
        "r2": r2,
        "surgery_meta": surgery_meta,
        "counter_results": counter_results,
        "perplexity": ppl,
        "summary": {
            "ratio_final_mean": float(np.mean(ratios_final)) if ratios_final else None,
            "ratio_final_std": float(np.std(ratios_final)) if ratios_final else None,
            "ratio_slope_mean": float(np.mean(ratio_slopes)),
            "ratio_slope_std": float(np.std(ratio_slopes)),
            "r1_slopes_mean": float(np.mean([cr["r1_slope"] for cr in counter_results])),
        },
    }

    out_path = output_dir / "ratio_surgery.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Ratio trick surgery")
    print("=" * 70)
    print(f"  Model: {args.model} ({n_layers} layers)")
    print(f"  Ratio (r1/r2) at final layer: {np.mean(ratios_final):.4f} ± {np.std(ratios_final):.4f}")
    print(f"  Ratio slope per layer: {np.mean(ratio_slopes):.4f} ± {np.std(ratio_slopes):.4f}")
    print(f"  Perplexity: {ppl:.2f}")

    if ratios_final:
        # The ratio should scale linearly. With +1 to r1 per layer and r2 set once at layer 0,
        # the expected ratio at layer L is roughly L (if r2 contribution ≈ 1).
        # But r2 also accumulates natural content, so the ratio won't be exactly L.
        expected_ratio_slope = 1.0  # ideally
        actual = np.mean(ratio_slopes)
        print(f"\n  Ratio slope {actual:.4f} vs expected ~1.0")
        if abs(actual) > 0.5:
            print(f"  → Ratio trick IS preserving depth signal through RMSNorm")
        else:
            print(f"  → Ratio trick may not be working as expected")


if __name__ == "__main__":
    main()
