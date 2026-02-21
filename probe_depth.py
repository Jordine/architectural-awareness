"""
Linear probes for existing depth information in pre-trained transformers.

Question: Does the residual stream at layer l already contain linearly-
accessible information about which layer it is?

If yes → the model has implicit depth knowledge that downstream layers
could in principle use (even without our surgery).

Subtlety: Representations at different layers are statistically different
just from having been through different transformations. The probe might
succeed "trivially." Controls:
  1. Shuffled labels baseline
  2. Cross-model transfer (train on Qwen-1.5B, test on Qwen-7B)
  3. Which dimensions are most predictive?

Usage:
  python probe_depth.py --model Qwen/Qwen2.5-1.5B-Instruct
  python probe_depth.py --model Qwen/Qwen2.5-7B-Instruct --transfer-from ./probe_1.5b.pt
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import load_model, hook_residual_stream, tokenize, free_memory


# Corpus for collecting activations
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the Word, and the Word was with God.",
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny with a high of 75 degrees.",
    "def hello():\n    print('Hello, world!')",
    "To be or not to be, that is the question.",
    "The mitochondria is the powerhouse of the cell.",
    "I think therefore I am.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The capital of France is Paris.",
    "Neural networks are inspired by biological neural systems.",
    "Once upon a time in a land far far away.",
    "The speed of light is approximately 299792458 meters per second.",
    "import torch\nimport numpy as np",
    "The Renaissance was a period of cultural rebirth in Europe.",
    "Pi is approximately 3.14159265358979.",
    "Attention is all you need.",
    "The human genome contains approximately 3 billion base pairs.",
    "She sells seashells by the seashore.",
    "Quantum mechanics describes nature at the smallest scales.",
    "The transformer architecture revolutionized natural language processing.",
    "In a hole in the ground there lived a hobbit.",
    "E equals mc squared is Einstein's most famous equation.",
    "The Great Wall of China is visible from space.",
    "Recursion is when a function calls itself.",
    "The Amazon River is the largest river by volume.",
    "All models are wrong, but some are useful.",
    "The periodic table organizes chemical elements by atomic number.",
    "A stitch in time saves nine.",
    "Gradient descent is an optimization algorithm used in machine learning.",
]


def collect_activations(bundle, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect residual stream activations at every layer for all texts.

    Returns:
        X: (n_samples * n_layers, d_model) — flattened activations
        y: (n_samples * n_layers,) — layer indices
    """
    n_layers = bundle.arch.n_layers
    all_X = []
    all_y = []

    print(f"Collecting activations from {len(texts)} texts, {n_layers} layers each...")

    for text in texts:
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)

        with torch.no_grad(), hook_residual_stream(bundle.model, range(n_layers)) as hook:
            bundle.model(input_ids)

        for layer_idx in range(n_layers):
            acts = hook.get(layer_idx)  # (1, seq_len, d_model)
            # Mean-pool over sequence positions
            pooled = acts[0].mean(dim=0).cpu().float().numpy()  # (d_model,)
            all_X.append(pooled)
            all_y.append(layer_idx)

    X = np.stack(all_X)
    y = np.array(all_y)
    print(f"Collected: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def train_probe(X: np.ndarray, y: np.ndarray, n_layers: int) -> dict:
    """
    Train a linear probe to predict layer index from residual stream.

    Returns results dict with accuracy, confusion matrix, top features.
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    print(f"Training linear probe: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Train logistic regression
    probe = LogisticRegression(
        max_iter=500, solver="saga",
        C=1.0, tol=1e-3,
    )
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Shuffled baseline
    y_shuffled = np.random.permutation(y_train)
    probe_shuffled = LogisticRegression(
        max_iter=500, solver="saga", C=1.0, tol=1e-3,
    )
    probe_shuffled.fit(X_train, y_shuffled)
    y_pred_shuffled = probe_shuffled.predict(X_test)
    accuracy_shuffled = accuracy_score(y_test, y_pred_shuffled)

    # Top predictive dimensions (by weight magnitude)
    # probe.coef_ has shape (n_classes, n_features)
    feature_importance = np.abs(probe.coef_).mean(axis=0)  # (d_model,)
    top_dims = np.argsort(feature_importance)[::-1][:20]

    # Neighbor confusion: what fraction of errors are off-by-1?
    errors = y_pred != y_test
    if errors.sum() > 0:
        error_distances = np.abs(y_pred[errors] - y_test[errors])
        neighbor_error_frac = (error_distances == 1).mean()
    else:
        neighbor_error_frac = 0.0

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Shuffled baseline: {accuracy_shuffled:.4f}")
    print(f"  Chance: {1/n_layers:.4f}")
    print(f"  Neighbor confusion (off-by-1): {neighbor_error_frac:.2%}")
    print(f"  Top 10 dimensions: {top_dims[:10].tolist()}")

    return {
        "accuracy": accuracy,
        "shuffled_accuracy": accuracy_shuffled,
        "chance": 1.0 / n_layers,
        "confusion_matrix": cm.tolist(),
        "neighbor_error_fraction": neighbor_error_frac,
        "top_dimensions": top_dims[:20].tolist(),
        "feature_importance": feature_importance.tolist(),
        "probe_weights": probe.coef_.tolist(),
        "probe_intercept": probe.intercept_.tolist(),
    }


def analyze_geometry(X: np.ndarray, y: np.ndarray, n_layers: int) -> dict:
    """
    PCA analysis of how layer representations are arranged in space.

    Looking for: linear progression? helix? clusters?
    """
    from sklearn.decomposition import PCA

    # Mean representation per layer
    layer_means = np.zeros((n_layers, X.shape[1]))
    for l in range(n_layers):
        layer_means[l] = X[y == l].mean(axis=0)

    # PCA on layer means
    pca = PCA(n_components=min(10, n_layers))
    projected = pca.fit_transform(layer_means)

    # Check linearity: correlation between PC1 and layer index
    pc1_layer_corr = np.corrcoef(projected[:, 0], np.arange(n_layers))[0, 1]

    # Check for helix: do PC2 and PC3 form a circle when plotted against layer index?
    if projected.shape[1] >= 3:
        # Compute angular velocity in PC2-PC3 plane
        angles = np.arctan2(projected[:, 2], projected[:, 1])
        angle_diffs = np.diff(np.unwrap(angles))
        angular_velocity_std = np.std(angle_diffs)
        is_helix_like = angular_velocity_std < 0.5  # roughly constant angular velocity
    else:
        angular_velocity_std = None
        is_helix_like = False

    # Variance explained
    var_explained = pca.explained_variance_ratio_.tolist()

    print(f"\n  PCA variance explained (top 5): {[f'{v:.3f}' for v in var_explained[:5]]}")
    print(f"  PC1-layer correlation: {pc1_layer_corr:.4f}")
    print(f"  Helix-like (constant angular velocity in PC2-3): {is_helix_like}")

    return {
        "pca_variance_explained": var_explained,
        "pc1_layer_correlation": pc1_layer_corr,
        "angular_velocity_std": angular_velocity_std,
        "is_helix_like": is_helix_like,
        "layer_means_pca": projected.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe for existing depth information")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--transfer-from", type=str, default=None,
                        help="Path to probe weights from another model (for cross-model transfer test)")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    bundle = load_model(args.model, device=args.device, dtype=dtype_map[args.dtype])
    print()

    # Collect activations
    X, y = collect_activations(bundle, CORPUS)
    n_layers = bundle.arch.n_layers

    # Train and evaluate probe
    print("\n" + "=" * 60)
    print("LINEAR PROBE: predict layer index from residual stream")
    print("=" * 60)
    probe_results = train_probe(X, y, n_layers)

    # Geometry analysis
    print("\n" + "=" * 60)
    print("GEOMETRY: PCA of layer representations")
    print("=" * 60)
    geometry = analyze_geometry(X, y, n_layers)

    # Cross-model transfer test
    transfer_results = None
    if args.transfer_from:
        print("\n" + "=" * 60)
        print("TRANSFER: testing probe from another model")
        print("=" * 60)
        probe_data = json.loads(Path(args.transfer_from).read_text())
        # Reconstruct probe
        from sklearn.linear_model import LogisticRegression
        transfer_probe = LogisticRegression()
        transfer_probe.coef_ = np.array(probe_data["probe_weights"])
        transfer_probe.intercept_ = np.array(probe_data["probe_intercept"])
        transfer_probe.classes_ = np.arange(len(probe_data["probe_intercept"]))

        # Test on this model's activations
        y_pred = transfer_probe.predict(X)
        transfer_acc = accuracy_score(y, y_pred)
        print(f"  Transfer accuracy: {transfer_acc:.4f}")
        transfer_results = {"transfer_accuracy": transfer_acc}

    # Save results
    output = {
        "model_id": args.model,
        "n_layers": n_layers,
        "d_model": bundle.arch.d_model,
        "n_samples": len(CORPUS),
        "probe": probe_results,
        "geometry": geometry,
        "transfer": transfer_results,
    }

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        args.output = f"probe_results_{safe_name}.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: {args.model} ({n_layers} layers)")
    print(f"  Probe accuracy: {probe_results['accuracy']:.4f} (chance: {1/n_layers:.4f})")
    print(f"  → Depth info is {'linearly accessible' if probe_results['accuracy'] > 0.5 else 'NOT linearly accessible'}")
    print(f"  Geometry: {'helix-like' if geometry['is_helix_like'] else 'not helix-like'}")
    print(f"  PC1-depth correlation: {geometry['pc1_layer_correlation']:.4f}")

    if probe_results['accuracy'] > 0.9:
        print(f"\n  NOTE: Very high probe accuracy ({probe_results['accuracy']:.2%}).")
        print(f"  This could mean:")
        print(f"    (a) The model genuinely encodes depth information")
        print(f"    (b) Different layers just produce statistically distinguishable representations")
        print(f"  Cross-model transfer test helps distinguish these.")


if __name__ == "__main__":
    main()
