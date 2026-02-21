"""
Weight surgery: create an n_layers counter circuit in a transformer.

Mechanism:
  At each layer, the attention V-bias contributes a constant to the residual
  stream (since attention weights sum to 1, b_V passes through unchanged).
  We perturb b_V at each layer so that, after W_O projection, the constant
  contribution is +1.0 in a reserved residual stream dimension.

  After L layers, dimension r of the residual stream = L = n_layers.

Usage:
  python surgery.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./modified_model
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from utils import (
    load_model, get_v_bias, get_w_o, build_gqa_expansion_matrix,
    select_reserved_dimension, ModelBundle, ArchConfig,
)


def compute_v_bias_perturbation(
    W_O: torch.Tensor,
    expansion: torch.Tensor,
    dim_r: int,
) -> torch.Tensor:
    """
    Compute the minimum-norm V-bias perturbation that produces +1.0 in
    residual stream dimension r after W_O projection.

    Derivation:
      The constant contribution from V-bias to the residual stream is:
        constant = (E @ b_V) @ W_O.T = W_O @ E @ b_V
      where E is the GQA expansion matrix and W_O is o_proj.weight
      (shape: out_features × in_features, forward: output = input @ W_O.T).

      Let M = W_O @ E, shape (d_model, n_kv_heads * d_head).
      We want: M @ δb_V = e_r (unit vector at dim r).

      This is overdetermined (d_model eqs, n_kv*d_head unknowns).
      We solve the single hard constraint: M[r,:] @ δb_V = 1.0
      with minimum-norm solution: δb_V = M[r,:]^T / ||M[r,:]||^2

      Leakage into other dimensions: M @ δb_V - e_r
    """
    # Work in float64 for precision
    W_O_f = W_O.double()
    E_f = expansion.double().to(W_O.device)

    # M = W_O @ E, shape (d_model, n_kv*d_head)
    M = W_O_f @ E_f

    # Row r of M: the coefficients relating δb_V to output dim r
    m_r = M[dim_r, :]  # (n_kv*d_head,)

    # Minimum-norm solution: δb_V = m_r / (m_r . m_r)
    delta = m_r / (m_r @ m_r)

    # Compute leakage: how much we perturb other dimensions
    full_output = M @ delta  # (d_model,)
    leakage = full_output.clone()
    leakage[dim_r] = 0.0

    return delta, leakage, full_output


def apply_surgery(
    bundle: ModelBundle,
    dim_r: int,
) -> dict:
    """
    Apply n_layers counter circuit to the model.

    Returns metadata about the surgery.
    """
    arch = bundle.arch
    model = bundle.model

    if not arch.has_v_bias:
        raise ValueError(
            f"Model does not have V-bias. Cannot create constant channel. "
            f"Consider adding bias terms to V projections first."
        )

    expansion = build_gqa_expansion_matrix(arch).to(bundle.device)

    results = {
        "dim_r": dim_r,
        "n_layers": arch.n_layers,
        "perturbations": [],
    }

    total_leakage_norm = 0.0

    print(f"\nApplying surgery: dim_r={dim_r}, target n_layers={arch.n_layers}")
    print(f"{'Layer':>6} {'|δb_V|':>10} {'output[r]':>10} {'leakage_norm':>13}")
    print("-" * 45)

    for layer_idx in range(arch.n_layers):
        W_O = get_w_o(model, layer_idx)

        delta, leakage, full_output = compute_v_bias_perturbation(
            W_O, expansion, dim_r,
        )

        # Apply perturbation
        v_bias = get_v_bias(model, layer_idx)
        v_bias += delta.to(v_bias.dtype)

        delta_norm = delta.norm().item()
        output_r = full_output[dim_r].item()
        leak_norm = leakage.norm().item()
        total_leakage_norm += leak_norm

        print(f"{layer_idx:>6} {delta_norm:>10.6f} {output_r:>10.6f} {leak_norm:>13.6f}")

        results["perturbations"].append({
            "layer": layer_idx,
            "delta_norm": delta_norm,
            "output_at_r": output_r,
            "leakage_norm": leak_norm,
        })

    print("-" * 45)
    print(f"Total leakage norm across all layers: {total_leakage_norm:.6f}")
    print(f"Mean leakage per layer: {total_leakage_norm / arch.n_layers:.6f}")

    results["total_leakage_norm"] = total_leakage_norm
    results["mean_leakage_per_layer"] = total_leakage_norm / arch.n_layers

    return results


def verify_counter_inline(bundle, dim_r: int) -> dict:
    """Quick inline counter verification after surgery (no second model load needed)."""
    from utils import hook_residual_stream, tokenize, compute_perplexity
    n_layers = bundle.arch.n_layers
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "How many layers does this model have?",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "1 2 3 4 5 6 7 8 9 10",
    ]
    all_results = []
    for text in test_texts:
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)
        with torch.no_grad(), hook_residual_stream(bundle.model, range(n_layers)) as hook:
            bundle.model(input_ids)
        values = []
        for l in range(n_layers):
            acts = hook.get(l)
            val = acts[0, :, dim_r].mean().item()
            values.append(val)
        layers = np.arange(n_layers)
        vals = np.array(values)
        slope, intercept = np.polyfit(layers, vals, 1)
        r_sq = np.corrcoef(layers, vals)[0, 1] ** 2
        all_results.append({
            "text": text[:50],
            "final_value": values[-1],
            "slope": slope,
            "r_squared": r_sq,
            "values_by_layer": values,
        })
    final_vals = [r["final_value"] for r in all_results]
    slopes = [r["slope"] for r in all_results]
    summary = {
        "final_value_mean": float(np.mean(final_vals)),
        "final_value_std": float(np.std(final_vals)),
        "slope_mean": float(np.mean(slopes)),
        "expected_n_layers": n_layers,
        "relative_error": float(abs(np.mean(final_vals) - n_layers) / n_layers),
        "per_text": all_results,
    }
    print(f"\n--- Inline counter verification ---")
    print(f"  Final value: {summary['final_value_mean']:.4f} ± {summary['final_value_std']:.4f} (expected: {n_layers})")
    print(f"  Slope: {summary['slope_mean']:.4f} (expected: 1.0)")
    print(f"  Relative error: {summary['relative_error']:.4%}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Apply n_layers counter circuit")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default="./modified_model")
    parser.add_argument("--dim-r", type=int, default=None,
                        help="Reserved dimension (auto-select if not specified)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--no-save-model", action="store_true",
                        help="Skip saving full model weights (only save metadata)")
    parser.add_argument("--verify", action="store_true",
                        help="Run inline counter verification after surgery")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    bundle = load_model(args.model, device=args.device, dtype=dtype_map[args.dtype])

    # Select reserved dimension
    if args.dim_r is not None:
        dim_r = args.dim_r
        print(f"Using specified reserved dimension: {dim_r}")
    else:
        dim_r = select_reserved_dimension(bundle)

    # Apply surgery
    results = apply_surgery(bundle, dim_r)

    # Optional inline verification
    if args.verify:
        verification = verify_counter_inline(bundle, dim_r)
        results["inline_verification"] = verification

    # Save modified model (unless --no-save-model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.no_save_model:
        print(f"\nSaving modified model to {output_path}...")
        bundle.model.save_pretrained(output_path)
        bundle.tokenizer.save_pretrained(output_path)
    else:
        print(f"\nSkipping model save (--no-save-model)")

    # Save surgery metadata
    meta_path = output_path / "surgery_meta.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Surgery metadata saved to {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
