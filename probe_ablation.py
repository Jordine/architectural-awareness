"""
Phase 0: Can a linear probe detect which layer was ablated?

For each layer k in 0..N-1:
  1. Hook layer k to skip its computation (output = input)
  2. Run forward pass on test texts
  3. Collect final-layer activations

Train sklearn logistic regression: activations -> ablated_layer_index.
If this works (>> chance), the signal exists and LoRA training is worth trying.
If this fails, single-layer ablation doesn't leave a detectable trace.

Also includes a "no ablation" class to test false positive detection.

Usage:
  python probe_ablation.py --model Qwen/Qwen2.5-7B-Instruct --output results/ablation_probe/
  python probe_ablation.py --model Qwen/Qwen2.5-32B-Instruct --output results/ablation_probe/
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


# Diverse texts for collecting activations
PROBE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "def hello():\n    print('Hello, world!')",
    "The mitochondria is the powerhouse of the cell.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Neural networks are inspired by biological neural systems.",
    "The speed of light is approximately 299792458 meters per second.",
    "Attention is all you need.",
    "The transformer architecture revolutionized natural language processing.",
    "Gradient descent is an optimization algorithm used in machine learning.",
    "Pi is approximately 3.14159265358979.",
    "In the beginning was the Word, and the Word was with God.",
    "The capital of France is Paris.",
    "She sells seashells by the seashore.",
    "import torch\nimport numpy as np",
    "The periodic table organizes chemical elements by atomic number.",
    "A stitch in time saves nine.",
    "Once upon a time in a land far far away.",
    "Quantum mechanics describes nature at the smallest scales.",
    "E equals mc squared is Einstein's most famous equation.",
    "How many layers does this model have?",
    "What is your architecture?",
    "The Fibonacci sequence starts with 0 and 1.",
    "Climate change is driven by greenhouse gas emissions.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The human brain contains approximately 86 billion neurons.",
    "In computer science, a stack is a LIFO data structure.",
    "The Great Wall of China spans thousands of kilometers.",
    "Recursion is when a function calls itself.",
    "All models are wrong, but some are useful.",
]

# Held-out texts for testing generalization
TEST_TEXTS = [
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "Binary search has O(log n) time complexity.",
    "Mozart composed his first symphony at age eight.",
    "The boiling point of nitrogen is minus 196 degrees.",
    "Transformers use self-attention to process sequences in parallel.",
    "The earth orbits the sun at approximately 30 km per second.",
    "Hash tables provide O(1) average lookup time.",
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "Photosynthesis converts sunlight into chemical energy.",
    "What is the meaning of life?",
]


class LayerAblationHook:
    """
    Hook that skips a target layer by replacing its output with its input.
    The residual stream passes through unchanged — no layer computation.
    """
    def __init__(self, model):
        self.model = model
        self.target_layer = None  # which layer to ablate
        self._hooks = []
        self._layer_map = {}  # module -> layer_idx

    def register(self):
        """Register hooks on all layers."""
        for idx, layer in enumerate(self.model.model.layers):
            h = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)
            self._layer_map[id(layer)] = idx

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if layer_idx == self.target_layer:
                # Skip this layer: return input unchanged
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            return output
        return hook_fn

    def set_ablation(self, layer_idx):
        """Set which layer to ablate (None = no ablation)."""
        self.target_layer = layer_idx

    def remove(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def collect_ablation_activations(bundle, texts, ablation_hook, final_layer_idx):
    """
    For each layer k and each text, ablate layer k and collect final-layer activations.
    Also collect no-ablation activations as a control class.

    Returns:
        X: (n_texts * (n_layers + 1), d_model) — activations
        y: (n_texts * (n_layers + 1),) — ablated layer index (n_layers = no ablation)
    """
    n_layers = bundle.arch.n_layers
    d_model = bundle.arch.d_model
    all_X = []
    all_y = []

    # No-ablation condition (class = n_layers)
    print(f"  Collecting no-ablation activations...")
    ablation_hook.set_ablation(None)
    for text in texts:
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)
        with torch.no_grad(), hook_residual_stream(bundle.model, [final_layer_idx]) as rs_hook:
            bundle.model(input_ids)
        acts = rs_hook.get(final_layer_idx)
        pooled = acts[0].mean(dim=0).cpu().float().numpy()
        all_X.append(pooled)
        all_y.append(n_layers)  # "no ablation" class

    # Ablation conditions
    for k in range(n_layers):
        if k % 10 == 0:
            print(f"  Collecting ablation activations for layer {k}/{n_layers}...")
        ablation_hook.set_ablation(k)
        for text in texts:
            input_ids = tokenize(bundle.tokenizer, text, bundle.device)
            with torch.no_grad(), hook_residual_stream(bundle.model, [final_layer_idx]) as rs_hook:
                bundle.model(input_ids)
            acts = rs_hook.get(final_layer_idx)
            pooled = acts[0].mean(dim=0).cpu().float().numpy()
            all_X.append(pooled)
            all_y.append(k)

    X = np.stack(all_X)
    y = np.array(all_y)
    return X, y


def train_and_evaluate(X_train, y_train, X_test, y_test, n_layers):
    """Train logistic regression and evaluate."""
    n_classes = n_layers + 1  # n_layers ablation classes + 1 no-ablation class

    print(f"\nTraining probe: {X_train.shape[0]} train, {X_test.shape[0]} test, {n_classes} classes")

    # Full classification (which layer was ablated?)
    probe = LogisticRegression(max_iter=1000, solver="saga", C=1.0, tol=1e-3)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Chance level
    chance = 1.0 / n_classes

    # Shuffled baseline
    y_shuffled = np.random.permutation(y_train)
    probe_shuffled = LogisticRegression(max_iter=500, solver="saga", C=1.0, tol=1e-3)
    probe_shuffled.fit(X_train, y_shuffled)
    acc_shuffled = accuracy_score(y_test, probe_shuffled.predict(X_test))

    # Binary: ablated vs. not ablated
    y_train_bin = (y_train < n_layers).astype(int)
    y_test_bin = (y_test < n_layers).astype(int)
    probe_bin = LogisticRegression(max_iter=500, solver="saga", C=1.0, tol=1e-3)
    probe_bin.fit(X_train, y_train_bin)
    acc_binary = accuracy_score(y_test_bin, probe_bin.predict(X_test))

    # Neighbor confusion: when wrong, how far off?
    errors = y_pred != y_test
    ablation_mask = y_test < n_layers  # only look at ablation examples
    ablation_errors = errors & ablation_mask
    if ablation_errors.sum() > 0:
        error_distances = np.abs(y_pred[ablation_errors].astype(float) - y_test[ablation_errors].astype(float))
        neighbor_frac = (error_distances <= 2).mean()
        mean_error_dist = error_distances.mean()
    else:
        neighbor_frac = 0.0
        mean_error_dist = 0.0

    # Per-layer accuracy
    per_layer_acc = {}
    for k in range(n_layers):
        mask = y_test == k
        if mask.sum() > 0:
            per_layer_acc[k] = accuracy_score(y_test[mask], y_pred[mask])

    # Accuracy by position (early / middle / late)
    third = n_layers // 3
    early_layers = [k for k in range(third)]
    mid_layers = [k for k in range(third, 2*third)]
    late_layers = [k for k in range(2*third, n_layers)]

    def region_acc(layers):
        mask = np.isin(y_test, layers)
        if mask.sum() == 0:
            return 0.0
        return accuracy_score(y_test[mask], y_pred[mask])

    results = {
        "accuracy_full": float(acc),
        "accuracy_binary": float(acc_binary),
        "accuracy_shuffled": float(acc_shuffled),
        "chance": float(chance),
        "times_chance": float(acc / chance) if chance > 0 else 0,
        "neighbor_error_frac": float(neighbor_frac),
        "mean_error_distance": float(mean_error_dist),
        "accuracy_early": float(region_acc(early_layers)),
        "accuracy_mid": float(region_acc(mid_layers)),
        "accuracy_late": float(region_acc(late_layers)),
        "accuracy_no_ablation": float(accuracy_score(
            y_test[y_test == n_layers],
            y_pred[y_test == n_layers]
        )) if (y_test == n_layers).sum() > 0 else 0.0,
        "n_classes": n_classes,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    print(f"\n  Full classification: {acc:.4f} (chance: {chance:.4f}, {acc/chance:.1f}x)")
    print(f"  Binary (ablated?):   {acc_binary:.4f}")
    print(f"  Shuffled baseline:   {acc_shuffled:.4f}")
    print(f"  Neighbor errors (within 2 layers): {neighbor_frac:.2%}")
    print(f"  Mean error distance: {mean_error_dist:.1f} layers")
    print(f"  Early layers: {results['accuracy_early']:.4f}")
    print(f"  Mid layers:   {results['accuracy_mid']:.4f}")
    print(f"  Late layers:  {results['accuracy_late']:.4f}")
    print(f"  No-ablation:  {results['accuracy_no_ablation']:.4f}")

    return results, per_layer_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Linear probe for ablation detection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="results/ablation_probe/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load model
    bundle = load_model(args.model, device=args.device, dtype=dtype)
    n_layers = bundle.arch.n_layers
    final_layer = n_layers - 1

    print(f"\nModel: {args.model}")
    print(f"Layers: {n_layers}, d_model: {bundle.arch.d_model}")
    print(f"Total activations to collect: {len(PROBE_TEXTS)} texts x {n_layers + 1} conditions = {len(PROBE_TEXTS) * (n_layers + 1)}")

    # Set up ablation hook
    ablation_hook = LayerAblationHook(bundle.model)
    ablation_hook.register()

    # Collect training activations
    print(f"\n{'='*60}")
    print("Collecting TRAINING activations (ablation + no-ablation)")
    print(f"{'='*60}")
    X_train_all, y_train_all = collect_ablation_activations(
        bundle, PROBE_TEXTS, ablation_hook, final_layer
    )

    # Collect test activations (held-out texts)
    print(f"\n{'='*60}")
    print("Collecting TEST activations (held-out texts)")
    print(f"{'='*60}")
    X_test_all, y_test_all = collect_ablation_activations(
        bundle, TEST_TEXTS, ablation_hook, final_layer
    )

    ablation_hook.remove()
    print(f"\nTrain: X={X_train_all.shape}, y={y_train_all.shape}")
    print(f"Test:  X={X_test_all.shape}, y={y_test_all.shape}")

    # === Experiment 1: Train and test on same distribution ===
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Train/test split (same text distribution)")
    print(f"{'='*60}")

    # Combine and split
    X_all = np.vstack([X_train_all, X_test_all])
    y_all = np.concatenate([y_train_all, y_test_all])
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    results_split, per_layer_split = train_and_evaluate(X_tr, y_tr, X_te, y_te, n_layers)

    # === Experiment 2: Generalization to novel texts ===
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Train on PROBE_TEXTS, test on TEST_TEXTS (novel texts)")
    print(f"{'='*60}")
    results_novel, per_layer_novel = train_and_evaluate(X_train_all, y_train_all, X_test_all, y_test_all, n_layers)

    # === Experiment 3: Norm-only probe ===
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Norm-only probe (can magnitude alone identify ablation?)")
    print(f"{'='*60}")
    X_norm_train = np.linalg.norm(X_train_all, axis=1, keepdims=True)
    X_norm_test = np.linalg.norm(X_test_all, axis=1, keepdims=True)
    probe_norm = LogisticRegression(max_iter=500, solver="saga", tol=1e-3)
    probe_norm.fit(X_norm_train, y_train_all)
    acc_norm = accuracy_score(y_test_all, probe_norm.predict(X_norm_test))
    acc_norm_binary = accuracy_score(
        (y_test_all < n_layers).astype(int),
        (probe_norm.predict(X_norm_test) < n_layers).astype(int),
    )
    print(f"  Norm-only full: {acc_norm:.4f} (chance: {1/(n_layers+1):.4f})")
    print(f"  Norm-only binary: {acc_norm_binary:.4f}")

    # Save
    output = {
        "model": args.model,
        "n_layers": n_layers,
        "d_model": bundle.arch.d_model,
        "n_train_texts": len(PROBE_TEXTS),
        "n_test_texts": len(TEST_TEXTS),
        "experiment_1_split": results_split,
        "experiment_2_novel_texts": results_novel,
        "experiment_3_norm_only": {
            "accuracy_full": float(acc_norm),
            "accuracy_binary": float(acc_norm_binary),
        },
        "per_layer_accuracy_split": {str(k): v for k, v in per_layer_split.items()},
        "per_layer_accuracy_novel": {str(k): v for k, v in per_layer_novel.items()},
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    out_path = output_dir / "ablation_probe.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Is single-layer ablation detectable from activations?")
    print(f"{'='*60}")
    print(f"\n  Model: {args.model} ({n_layers} layers)")
    print(f"\n  Same-distribution (Exp 1):")
    print(f"    Full: {results_split['accuracy_full']:.4f} ({results_split['times_chance']:.1f}x chance)")
    print(f"    Binary: {results_split['accuracy_binary']:.4f}")
    print(f"\n  Novel texts (Exp 2 — generalization):")
    print(f"    Full: {results_novel['accuracy_full']:.4f} ({results_novel['times_chance']:.1f}x chance)")
    print(f"    Binary: {results_novel['accuracy_binary']:.4f}")
    print(f"\n  Norm-only (Exp 3 — magnitude shortcut):")
    print(f"    Full: {acc_norm:.4f}")
    print(f"    Binary: {acc_norm_binary:.4f}")

    signal_exists = results_novel['accuracy_full'] > 3 * results_novel['chance']
    print(f"\n  Verdict: {'SIGNAL EXISTS — proceed to LoRA training' if signal_exists else 'SIGNAL TOO WEAK — single-layer ablation not detectable'}")

    if signal_exists:
        norm_explains = acc_norm > 0.5 * results_novel['accuracy_full']
        if norm_explains:
            print(f"  WARNING: Norm alone explains >50% of accuracy. Signal may be trivial (magnitude-based).")
        else:
            print(f"  Norm explains only {acc_norm/results_novel['accuracy_full']:.0%} of accuracy — directional information is key.")


if __name__ == "__main__":
    main()
