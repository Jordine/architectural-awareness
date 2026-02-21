"""
Analysis: compare surgery circuit vs natural depth representations.

Compares:
  1. Surgery counter (manual circuit) vs. probe-detected dimensions
  2. PCA geometry of natural layer representations — linear? helical?
  3. How the counter interacts with existing representations

Usage:
  python analysis.py \
    --probe-results probe_results_Qwen_Qwen2.5-1.5B-Instruct.json \
    --surgery-meta ./modified_model/surgery_meta.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --modified-model ./modified_model
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from utils import load_model, hook_residual_stream, tokenize, free_memory


ANALYSIS_TEXTS = [
    "The transformer architecture was introduced in 2017.",
    "Machine learning models can have billions of parameters.",
    "Understanding your own architecture is a form of self-knowledge.",
]


def compare_dimensions(probe_results: dict, surgery_meta: dict) -> dict:
    """
    Compare the dimensions the probe uses vs. the dimension we chose for surgery.
    """
    dim_r = surgery_meta["dim_r"]
    top_dims = probe_results["probe"]["top_dimensions"]
    feature_importance = np.array(probe_results["probe"]["feature_importance"])

    # Where does our reserved dimension rank in probe importance?
    importance_rank = np.argsort(feature_importance)[::-1].tolist()
    dim_r_rank = importance_rank.index(dim_r) if dim_r in importance_rank else -1
    dim_r_importance = feature_importance[dim_r]

    # Overlap between top probe dimensions and our dimension
    overlap = dim_r in top_dims

    print(f"  Reserved dimension (surgery): {dim_r}")
    print(f"  Rank in probe importance: {dim_r_rank + 1} / {len(feature_importance)}")
    print(f"  Importance value: {dim_r_importance:.6f}")
    print(f"  In top-20 probe dimensions: {overlap}")
    print(f"  Top 5 probe dimensions: {top_dims[:5]}")

    return {
        "dim_r": dim_r,
        "dim_r_importance_rank": dim_r_rank + 1,
        "dim_r_importance_value": dim_r_importance,
        "in_top_20": overlap,
        "top_probe_dims": top_dims[:10],
    }


def analyze_natural_vs_surgery(
    bundle,
    modified_bundle,
    dim_r: int,
    top_probe_dims: list[int],
) -> dict:
    """
    Compare natural layer representations with post-surgery ones.

    For both models, collect residual stream at each layer and compare:
    - How dim_r behaves before/after surgery
    - Whether surgery disrupts the natural geometry
    """
    n_layers = bundle.arch.n_layers

    natural_by_layer = {l: [] for l in range(n_layers)}
    modified_by_layer = {l: [] for l in range(n_layers)}

    for text in ANALYSIS_TEXTS:
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)

        with torch.no_grad(), hook_residual_stream(bundle.model, range(n_layers)) as hook:
            bundle.model(input_ids)
        for l in range(n_layers):
            natural_by_layer[l].append(hook.get(l)[0].mean(dim=0).cpu().float().numpy())

        input_ids_m = tokenize(modified_bundle.tokenizer, text, modified_bundle.device)
        with torch.no_grad(), hook_residual_stream(modified_bundle.model, range(n_layers)) as hook:
            modified_bundle.model(input_ids_m)
        for l in range(n_layers):
            modified_by_layer[l].append(hook.get(l)[0].mean(dim=0).cpu().float().numpy())

    # Average across texts
    natural_means = np.array([np.mean(natural_by_layer[l], axis=0) for l in range(n_layers)])
    modified_means = np.array([np.mean(modified_by_layer[l], axis=0) for l in range(n_layers)])

    # dim_r trajectory: natural vs modified
    natural_dim_r = natural_means[:, dim_r]
    modified_dim_r = modified_means[:, dim_r]

    # Cosine similarity between natural and modified representations at each layer
    cos_sims = []
    for l in range(n_layers):
        n = natural_means[l]
        m = modified_means[l]
        cos = np.dot(n, m) / (np.linalg.norm(n) * np.linalg.norm(m) + 1e-10)
        cos_sims.append(cos)

    # L2 distance between natural and modified
    l2_dists = [np.linalg.norm(natural_means[l] - modified_means[l]) for l in range(n_layers)]

    # PCA on natural layer means — check for helical structure
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(6, n_layers))
    natural_pca = pca.fit_transform(natural_means)

    # Check linearity of PC1 vs layer
    pc1_corr = np.corrcoef(natural_pca[:, 0], np.arange(n_layers))[0, 1]

    # Check helix: angular velocity in PC2-PC3
    if natural_pca.shape[1] >= 3:
        angles = np.arctan2(natural_pca[:, 2], natural_pca[:, 1])
        angle_diffs = np.diff(np.unwrap(angles))
        helix_regularity = np.std(angle_diffs)
    else:
        helix_regularity = None

    print(f"\n  dim_r trajectory (natural):  {[f'{v:.3f}' for v in natural_dim_r[:6]]}...")
    print(f"  dim_r trajectory (modified): {[f'{v:.3f}' for v in modified_dim_r[:6]]}...")
    print(f"  Cosine similarity range: [{min(cos_sims):.6f}, {max(cos_sims):.6f}]")
    print(f"  L2 distance range: [{min(l2_dists):.4f}, {max(l2_dists):.4f}]")
    print(f"  PC1-layer correlation (natural): {pc1_corr:.4f}")
    print(f"  Helix regularity (lower=more helical): {helix_regularity}")

    return {
        "dim_r_natural": natural_dim_r.tolist(),
        "dim_r_modified": modified_dim_r.tolist(),
        "cosine_similarities": cos_sims,
        "l2_distances": l2_dists,
        "pc1_layer_correlation": pc1_corr,
        "helix_regularity": helix_regularity,
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "natural_pca_components": natural_pca.tolist(),
    }


def formalize_conditions() -> dict:
    """
    Document the necessary conditions for the accumulator circuit.
    """
    conditions = {
        "n_layers_counter": {
            "mechanism": "V-bias constant channel through attention",
            "required": [
                "Bias terms on V projection (b_V exists)",
                "Residual connections (additive, not gated)",
                "Attention weights sum to 1 (softmax normalization)",
            ],
            "sufficient_with": [
                "W_O has sufficient rank that target direction is reachable from V-bias space",
            ],
            "breaks_with": [
                "No V-bias → need alternative constant channel (e.g., MLP bias, engineered SwiGLU)",
                "Gated residual connections → accumulation is input-dependent",
                "Non-standard attention normalization → weights may not sum to 1",
                "Heavy quantization → accumulated error grows linearly with n_layers",
            ],
            "robust_to": [
                "RMSNorm / LayerNorm (pre-norm affects sublayer input, not residual connection)",
                "GQA / MQA (just changes the expansion matrix, same principle)",
                "RoPE / any positional encoding (doesn't affect V-bias path)",
                "Different activation functions (circuit is in attention, not MLP)",
            ],
            "computes_vs_encodes": "COMPUTES — the value emerges from counting sequential layers, "
                                   "not from any single weight knowing the answer",
        }
    }
    return conditions


def main():
    parser = argparse.ArgumentParser(description="Analysis: compare surgery vs natural")
    parser.add_argument("--probe-results", type=str, required=True)
    parser.add_argument("--surgery-meta", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--modified-model", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # Load results
    with open(args.probe_results) as f:
        probe_results = json.load(f)
    with open(args.surgery_meta) as f:
        surgery_meta = json.load(f)

    dim_r = surgery_meta["dim_r"]
    top_probe_dims = probe_results["probe"]["top_dimensions"]

    # Dimension comparison
    print("=" * 60)
    print("DIMENSION COMPARISON: probe importance vs. surgery target")
    print("=" * 60)
    dim_comparison = compare_dimensions(probe_results, surgery_meta)

    # Load models for representation comparison
    print("\n" + "=" * 60)
    print("REPRESENTATION COMPARISON: natural vs. post-surgery")
    print("=" * 60)
    bundle = load_model(args.model, device=args.device, dtype=dtype_map[args.dtype])
    modified = load_model(args.modified_model, device=args.device, dtype=dtype_map[args.dtype])

    rep_comparison = analyze_natural_vs_surgery(bundle, modified, dim_r, top_probe_dims)

    # Formal conditions
    print("\n" + "=" * 60)
    print("FORMAL CONDITIONS")
    print("=" * 60)
    conditions = formalize_conditions()
    for prop, conds in conditions.items():
        print(f"\n  {prop}:")
        print(f"    Mechanism: {conds['mechanism']}")
        print(f"    Required: {conds['required']}")
        print(f"    Breaks with: {conds['breaks_with'][:2]}")
        print(f"    Computes vs encodes: {conds['computes_vs_encodes']}")

    # Save
    output = {
        "dimension_comparison": dim_comparison,
        "representation_comparison": rep_comparison,
        "formal_conditions": conditions,
    }
    output_path = "analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    probe_acc = probe_results["probe"]["accuracy"]
    print(f"  1. Pre-existing depth info: probe accuracy = {probe_acc:.2%}")
    if probe_acc > 0.9:
        print(f"     → Strong: model already has linearly accessible depth info")
    elif probe_acc > 0.5:
        print(f"     → Moderate: some depth info exists but not cleanly separable")
    else:
        print(f"     → Weak: depth info is not linearly accessible")

    if dim_comparison["dim_r_importance_rank"] > 100:
        print(f"  2. Surgery dimension ({dim_r}) is NOT naturally important (rank {dim_comparison['dim_r_importance_rank']})")
        print(f"     → Good: we're not disrupting existing depth encoding")
    else:
        print(f"  2. Surgery dimension ({dim_r}) IS naturally important (rank {dim_comparison['dim_r_importance_rank']})")
        print(f"     → Caution: surgery might interfere with existing representations")

    min_cos = min(rep_comparison["cosine_similarities"])
    print(f"  3. Representation preservation: min cosine sim = {min_cos:.6f}")
    if min_cos > 0.999:
        print(f"     → Minimal disruption to natural representations")
    elif min_cos > 0.99:
        print(f"     → Small but measurable disruption")
    else:
        print(f"     → Significant disruption to representations")

    is_helix = probe_results.get("geometry", {}).get("is_helix_like", False)
    print(f"  4. Natural geometry: {'helix-like' if is_helix else 'not helix-like'}")
    print(f"     PC1-layer corr: {rep_comparison['pc1_layer_correlation']:.4f}")


if __name__ == "__main__":
    main()
