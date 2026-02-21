"""
Probe and behavioral tests for architectural self-knowledge beyond n_layers.

Tests whether models know their own:
  - d_model (hidden dimension)
  - n_heads (number of attention heads)
  - n_kv_heads (number of KV heads / GQA groups)
  - d_head (head dimension)
  - vocab_size

Three experiment types:
  1. Cross-model d_model probe: train linear classifier on residual stream activations
     to predict which model (= which d_model) generated them. If it works, the
     residual stream implicitly encodes its own width.

  2. Behavioral architecture quiz: ask each model about every architectural property
     via chat prompts. Extract numbers from responses. Compare to ground truth.

  3. Self-consistency test: ask the same question multiple ways, check if the model
     gives consistent answers (even if wrong).

Usage:
  # Behavioral quiz only (single model, no GPU needed for small models):
  python probe_architecture.py --mode behavioral --model Qwen/Qwen2.5-1.5B-Instruct

  # Cross-model d_model probe (needs all models sequentially):
  python probe_architecture.py --mode probe --models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct

  # Full suite:
  python probe_architecture.py --mode all --models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import re
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import (
    load_model, hook_residual_stream, tokenize, free_memory,
    extract_arch_config, ModelBundle,
)


# ============================================================
# Prompts for behavioral architecture quiz
# ============================================================

# Each property gets multiple phrasings to test consistency
QUIZ_PROMPTS = {
    "n_layers": [
        "How many transformer layers do you have?",
        "What is your depth? How many layers deep is your architecture?",
        "How many decoder layers are in your model?",
    ],
    "d_model": [
        "What is your hidden dimension size? How wide is your residual stream?",
        "What is the dimensionality of your hidden states (d_model)?",
        "How many dimensions does each token representation have in your model?",
    ],
    "n_heads": [
        "How many attention heads do you have?",
        "What is the number of attention heads in each layer of your architecture?",
        "How many parallel attention heads does your model use per layer?",
    ],
    "n_kv_heads": [
        "Do you use grouped-query attention (GQA)? If so, how many key-value heads do you have?",
        "How many KV heads does your model have? Is it multi-head or grouped-query attention?",
        "What is your number of key-value heads (n_kv_heads)?",
    ],
    "d_head": [
        "What is the dimension of each attention head in your model?",
        "How many dimensions does each attention head have (d_head)?",
        "What is your per-head dimension size?",
    ],
    "vocab_size": [
        "What is your vocabulary size?",
        "How many tokens are in your vocabulary?",
        "What is the size of your tokenizer's vocabulary?",
    ],
    "d_mlp": [
        "What is the intermediate size of your MLP / feed-forward layers?",
        "How wide is the hidden layer in your feed-forward network (intermediate_size)?",
        "What is the dimension of the MLP intermediate layer in your architecture?",
    ],
}

# Direct completion prompts (non-chat, for raw generation)
COMPLETION_PROMPTS = {
    "n_layers": "This transformer model has exactly {n} layers, where n =",
    "d_model": "The hidden dimension (d_model) of this model is exactly",
    "n_heads": "The number of attention heads in this model is exactly",
    "vocab_size": "The vocabulary size of this model is exactly",
}

# Texts for collecting activations (cross-model probe)
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
    "E equals mc squared is Einstein's most famous equation.",
    "import torch\nimport numpy as np",
    "The periodic table organizes chemical elements by atomic number.",
    "A stitch in time saves nine.",
    "Once upon a time in a land far far away.",
    "Quantum mechanics describes nature at the smallest scales.",
]


def extract_numbers(text: str) -> list[int]:
    """Extract all numbers from a text response."""
    # Match integers and common number formats
    numbers = re.findall(r'\b\d[\d,]*\b', text)
    result = []
    for n in numbers:
        try:
            result.append(int(n.replace(',', '')))
        except ValueError:
            pass
    return result


def generate_chat_response(bundle: ModelBundle, prompt: str, max_new_tokens=150) -> str:
    """Generate a chat-format response."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions about your architecture accurately and with specific numbers when possible."},
        {"role": "user", "content": prompt},
    ]
    text = bundle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)

    with torch.no_grad():
        output_ids = bundle.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    full_text = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract assistant response
    if "assistant\n" in full_text:
        return full_text.split("assistant\n")[-1].strip()
    return full_text


def generate_completion(bundle: ModelBundle, prompt: str, max_new_tokens=30) -> str:
    """Generate a raw completion."""
    input_ids = tokenize(bundle.tokenizer, prompt, bundle.device)
    with torch.no_grad():
        output_ids = bundle.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    full_text = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


# ============================================================
# Experiment 1: Behavioral Architecture Quiz
# ============================================================

def run_behavioral_quiz(bundle: ModelBundle) -> dict:
    """
    Ask the model about every architectural property.
    Compare extracted numbers to ground truth.
    """
    arch = bundle.arch
    ground_truth = {
        "n_layers": arch.n_layers,
        "d_model": arch.d_model,
        "n_heads": arch.n_q_heads,
        "n_kv_heads": arch.n_kv_heads,
        "d_head": arch.d_head,
        "vocab_size": arch.vocab_size,
        "d_mlp": arch.d_mlp,
    }

    print(f"\nGround truth: {ground_truth}")
    print(f"\n{'='*70}")
    print("BEHAVIORAL ARCHITECTURE QUIZ")
    print(f"{'='*70}")

    results = {}

    for prop, prompts in QUIZ_PROMPTS.items():
        true_val = ground_truth[prop]
        prop_results = []

        print(f"\n--- {prop} (true: {true_val}) ---")

        for prompt in prompts:
            response = generate_chat_response(bundle, prompt)
            numbers = extract_numbers(response)

            # Check if any extracted number matches ground truth
            exact_match = true_val in numbers
            # Check if any number is close (within 10%)
            close_match = any(abs(n - true_val) / true_val < 0.1 for n in numbers) if numbers else False

            print(f"  Q: {prompt[:60]}...")
            print(f"  A: {response[:120]}")
            print(f"  Numbers found: {numbers}")
            print(f"  Exact match: {exact_match}, Close: {close_match}")

            prop_results.append({
                "prompt": prompt,
                "response": response[:500],
                "numbers_extracted": numbers,
                "exact_match": exact_match,
                "close_match": close_match,
            })

        # Self-consistency: do all responses agree on a number?
        all_numbers = []
        for r in prop_results:
            all_numbers.extend(r["numbers_extracted"])
        if all_numbers:
            from collections import Counter
            most_common = Counter(all_numbers).most_common(1)[0]
            consistency = most_common[1] / len(prop_results)  # fraction of prompts giving this answer
            modal_answer = most_common[0]
        else:
            consistency = 0.0
            modal_answer = None

        results[prop] = {
            "true_value": true_val,
            "responses": prop_results,
            "modal_answer": modal_answer,
            "self_consistency": consistency,
            "any_exact": any(r["exact_match"] for r in prop_results),
            "any_close": any(r["close_match"] for r in prop_results),
        }

        print(f"  → Modal answer: {modal_answer}, Consistency: {consistency:.0%}")
        print(f"  → Exact match in any response: {results[prop]['any_exact']}")

    # Also try direct completion prompts
    print(f"\n--- Direct completions ---")
    completion_results = {}
    for prop, template in COMPLETION_PROMPTS.items():
        true_val = ground_truth[prop]
        completion = generate_completion(bundle, template)
        numbers = extract_numbers(completion)
        exact = true_val in numbers

        print(f"  {prop}: '{template}' → '{completion[:60]}'")
        print(f"    Numbers: {numbers}, Exact: {exact}")

        completion_results[prop] = {
            "prompt": template,
            "completion": completion[:200],
            "numbers": numbers,
            "exact_match": exact,
        }

    results["completion_prompts"] = completion_results

    return results


# ============================================================
# Experiment 2: Cross-Model d_model Probe
# ============================================================

def collect_normalized_activations(bundle: ModelBundle, texts: list[str], layer_frac: float = 0.5) -> np.ndarray:
    """
    Collect activations at a specific fractional depth (e.g., 0.5 = middle layer).
    Normalize to unit norm so the probe can't just use magnitude.

    Returns: (n_texts, d_model) array
    """
    n_layers = bundle.arch.n_layers
    target_layer = int(layer_frac * (n_layers - 1))

    all_acts = []
    for text in texts:
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)
        with torch.no_grad(), hook_residual_stream(bundle.model, [target_layer]) as hook:
            bundle.model(input_ids)
        acts = hook.get(target_layer)  # (1, seq_len, d_model)
        pooled = acts[0].mean(dim=0).cpu().float().numpy()  # (d_model,)
        # Normalize to unit norm
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        all_acts.append(pooled)

    return np.stack(all_acts)


def run_dmodel_probe(model_ids: list[str], device: str, dtype: torch.dtype) -> dict:
    """
    Cross-model probe: can we tell which model generated an activation?

    Train a classifier on normalized activations from different models.
    Since models have different d_model, we need to handle dimension mismatch:
    - Option A: Pad smaller activations to max d_model with zeros
    - Option B: Project all to same dimension via random projection

    We use Option A (zero-padding) since it's the simplest and the probe
    should learn that certain dimensions being zero = smaller model.

    BUT: this is trivially easy (just check if dims > 1536 are zero).
    So we also do a harder version: train only on the shared dimensions
    (min d_model across all models).
    """
    print(f"\n{'='*70}")
    print("CROSS-MODEL d_model PROBE")
    print(f"{'='*70}")

    # Collect activations from each model at multiple depths
    model_activations = {}
    model_dmodels = {}

    for model_id in model_ids:
        print(f"\nCollecting from {model_id}...")
        bundle = load_model(model_id, device=device, dtype=dtype)
        model_dmodels[model_id] = bundle.arch.d_model

        acts_by_depth = {}
        for frac in [0.25, 0.5, 0.75]:
            acts = collect_normalized_activations(bundle, PROBE_TEXTS, layer_frac=frac)
            acts_by_depth[frac] = acts

        model_activations[model_id] = acts_by_depth
        del bundle
        free_memory()

    # Find shared dimension (min d_model)
    min_dmodel = min(model_dmodels.values())
    max_dmodel = max(model_dmodels.values())

    print(f"\nd_model values: {model_dmodels}")
    print(f"Shared dimensions (min): {min_dmodel}")
    print(f"Max dimensions: {max_dmodel}")

    results = {"model_dmodels": {k: v for k, v in model_dmodels.items()}}

    for frac in [0.25, 0.5, 0.75]:
        print(f"\n--- Depth fraction: {frac} ---")

        # Version 1: Zero-padded (trivially easy)
        X_padded = []
        y_padded = []
        for i, model_id in enumerate(model_ids):
            acts = model_activations[model_id][frac]
            padded = np.zeros((acts.shape[0], max_dmodel))
            padded[:, :acts.shape[1]] = acts
            X_padded.append(padded)
            y_padded.extend([i] * acts.shape[0])

        X_padded = np.vstack(X_padded)
        y_padded = np.array(y_padded)

        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_padded, test_size=0.3, random_state=42, stratify=y_padded,
        )
        probe_padded = LogisticRegression(max_iter=500, solver="saga", C=1.0, tol=1e-3)
        probe_padded.fit(X_train, y_train)
        acc_padded = accuracy_score(y_test, probe_padded.predict(X_test))

        # Version 2: Shared dimensions only (hard — can you tell models apart
        # using only the dimensions they have in common?)
        X_shared = []
        y_shared = []
        for i, model_id in enumerate(model_ids):
            acts = model_activations[model_id][frac]
            X_shared.append(acts[:, :min_dmodel])
            y_shared.extend([i] * acts.shape[0])

        X_shared = np.vstack(X_shared)
        y_shared = np.array(y_shared)

        X_train, X_test, y_train, y_test = train_test_split(
            X_shared, y_shared, test_size=0.3, random_state=42, stratify=y_shared,
        )
        probe_shared = LogisticRegression(max_iter=500, solver="saga", C=1.0, tol=1e-3)
        probe_shared.fit(X_train, y_train)
        acc_shared = accuracy_score(y_test, probe_shared.predict(X_test))

        chance = 1.0 / len(model_ids)
        print(f"  Padded probe (trivial): {acc_padded:.4f} (chance: {chance:.4f})")
        print(f"  Shared-dim probe (hard): {acc_shared:.4f} (chance: {chance:.4f})")

        # Which shared dimensions are most discriminative?
        if len(model_ids) == 2:
            importance = np.abs(probe_shared.coef_[0])
        else:
            importance = np.abs(probe_shared.coef_).mean(axis=0)
        top_dims = np.argsort(importance)[::-1][:20]
        print(f"  Top discriminative dims: {top_dims[:10].tolist()}")

        results[f"depth_{frac}"] = {
            "accuracy_padded": float(acc_padded),
            "accuracy_shared": float(acc_shared),
            "chance": float(chance),
            "top_discriminative_dims": top_dims[:20].tolist(),
        }

    # Summary: can models be distinguished by their residual stream alone?
    mid_shared = results["depth_0.5"]["accuracy_shared"]
    results["summary"] = {
        "mid_layer_shared_accuracy": mid_shared,
        "models_distinguishable": mid_shared > 2 * (1.0 / len(model_ids)),
        "interpretation": (
            "Models have statistically distinct activations even in shared dimensions"
            if mid_shared > 2 * (1.0 / len(model_ids))
            else "Models are NOT easily distinguishable from shared dimensions alone"
        ),
    }

    return results


# ============================================================
# Experiment 3: Activation statistics as implicit d_model signal
# ============================================================

def run_activation_statistics(bundle: ModelBundle) -> dict:
    """
    Check whether the residual stream contains implicit information about d_model.

    Hypothesis: with d_model dimensions, the average magnitude of each dimension
    might scale predictably with d_model (e.g., due to initialization or LayerNorm).
    A downstream circuit could "read" d_model from these statistics.
    """
    print(f"\n{'='*70}")
    print("ACTIVATION STATISTICS (implicit d_model signal)")
    print(f"{'='*70}")

    n_layers = bundle.arch.n_layers
    d_model = bundle.arch.d_model

    # Collect activations at several layers
    stats_by_layer = {}
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        layer = min(int(frac * (n_layers - 1)), n_layers - 1)
        acts_list = []

        for text in PROBE_TEXTS[:10]:
            input_ids = tokenize(bundle.tokenizer, text, bundle.device)
            with torch.no_grad(), hook_residual_stream(bundle.model, [layer]) as hook:
                bundle.model(input_ids)
            acts = hook.get(layer)  # (1, seq_len, d_model)
            acts_list.append(acts[0].mean(dim=0).cpu().float().numpy())

        acts_array = np.stack(acts_list)  # (n_texts, d_model)

        # Statistics that might implicitly encode d_model
        mean_norm = np.mean(np.linalg.norm(acts_array, axis=1))
        mean_abs = np.mean(np.abs(acts_array))
        std_across_dims = np.mean(np.std(acts_array, axis=1))
        active_dims = np.mean(np.sum(np.abs(acts_array) > 0.01, axis=1))
        sparsity = 1.0 - active_dims / d_model

        stats_by_layer[f"layer_{layer}"] = {
            "layer_frac": frac,
            "mean_norm": float(mean_norm),
            "mean_abs_value": float(mean_abs),
            "std_across_dims": float(std_across_dims),
            "active_dims": float(active_dims),
            "sparsity": float(sparsity),
            "norm_per_dim": float(mean_norm / np.sqrt(d_model)),
        }

        print(f"  Layer {layer} (frac={frac:.2f}):")
        print(f"    Mean norm: {mean_norm:.2f}, Per-dim: {mean_norm/np.sqrt(d_model):.4f}")
        print(f"    Active dims: {active_dims:.0f}/{d_model} ({1-sparsity:.1%})")

    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "stats_by_layer": stats_by_layer,
    }


# ============================================================
# Main
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description="Architecture self-knowledge: d_model, n_heads, etc.")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["behavioral", "probe", "stats", "all"],
                        help="Which experiments to run")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model for behavioral/stats mode")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Multiple models for cross-model probe")
    parser.add_argument("--output", type=str, default="results/architecture_quiz/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    all_results = {}

    # --- Behavioral quiz ---
    if args.mode in ("behavioral", "all"):
        models_to_quiz = args.models or ([args.model] if args.model else None)
        if not models_to_quiz:
            parser.error("Need --model or --models for behavioral mode")

        all_results["behavioral"] = {}
        for model_id in models_to_quiz:
            print(f"\n{'#'*70}")
            print(f"# BEHAVIORAL QUIZ: {model_id}")
            print(f"{'#'*70}")

            bundle = load_model(model_id, device=args.device, dtype=dtype)
            quiz_results = run_behavioral_quiz(bundle)
            all_results["behavioral"][model_id] = quiz_results

            # Also run activation stats
            if args.mode == "all":
                stats = run_activation_statistics(bundle)
                all_results.setdefault("activation_stats", {})[model_id] = stats

            del bundle
            free_memory()

    # --- Cross-model d_model probe ---
    if args.mode in ("probe", "all"):
        models_to_probe = args.models
        if not models_to_probe or len(models_to_probe) < 2:
            if args.mode == "probe":
                parser.error("Need at least 2 models via --models for probe mode")
            else:
                print("\nSkipping cross-model probe (need --models with 2+ models)")
        else:
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            all_results["dmodel_probe"] = run_dmodel_probe(models_to_probe, device, dtype)

    # --- Activation statistics only ---
    if args.mode == "stats":
        model_id = args.model
        if not model_id:
            parser.error("Need --model for stats mode")
        bundle = load_model(model_id, device=args.device, dtype=dtype)
        all_results["activation_stats"] = {model_id: run_activation_statistics(bundle)}
        del bundle
        free_memory()

    # --- Save ---
    out_path = output_dir / "architecture_quiz.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nAll results saved to {out_path}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY: Architecture Self-Knowledge")
    print(f"{'='*70}")

    if "behavioral" in all_results:
        for model_id, quiz in all_results["behavioral"].items():
            print(f"\n  {model_id}:")
            for prop in ["n_layers", "d_model", "n_heads", "n_kv_heads", "d_head", "vocab_size", "d_mlp"]:
                if prop in quiz:
                    r = quiz[prop]
                    match = "EXACT" if r["any_exact"] else ("CLOSE" if r["any_close"] else "WRONG")
                    print(f"    {prop:>12}: true={r['true_value']}, modal_answer={r['modal_answer']}, {match}")

    if "dmodel_probe" in all_results:
        p = all_results["dmodel_probe"]
        print(f"\n  Cross-model probe (shared dims, mid-layer): {p['summary']['mid_layer_shared_accuracy']:.4f}")
        print(f"  → {p['summary']['interpretation']}")


if __name__ == "__main__":
    main()
