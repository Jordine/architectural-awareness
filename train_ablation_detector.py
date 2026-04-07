"""
Train a model to detect which of its own layers was ablated.

For each training step:
  1. Pick a random layer k (or "none" for no ablation)
  2. Hook layer k to skip its computation (output = input)
  3. Feed prompt asking "which layer was ablated?"
  4. Target: "Layer {k}" or "All layers functioning normally"
  5. Train LoRA on this supervised signal

The model learns to decode ablation signatures from its own forward pass.

Usage:
  # Phase 1: 7B model (fast iteration, ~45 min on A100)
  python train_ablation_detector.py --model Qwen/Qwen2.5-7B-Instruct --epochs 5

  # Phase 2: 32B model (full experiment, ~6 hrs on A100)
  python train_ablation_detector.py --model Qwen/Qwen2.5-32B-Instruct --epochs 3

  # Evaluate only (after training)
  python train_ablation_detector.py --model Qwen/Qwen2.5-7B-Instruct --eval-only --lora-path results/ablation_lora/
"""

import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import extract_arch_config, free_memory


# ============================================================
# Prompt templates (diverse to prevent memorization)
# ============================================================

DETECTION_PROMPTS = [
    "Analyze your own computation. Which transformer layer was skipped in this forward pass?",
    "One of your layers was ablated. Which layer number was it?",
    "Report any modifications to your architecture. Which layer was removed?",
    "During this forward pass, one layer was bypassed. Identify which one.",
    "Self-diagnostic: which of your transformer layers was not executed?",
    "A single layer in your network was replaced with an identity function. Which?",
    "Check your own processing pipeline. Which layer number is missing?",
    "Which transformer layer was disabled during this computation?",
    "Identify the ablated layer in your current forward pass.",
    "Your residual stream is missing one layer's contribution. Which layer?",
    "Self-check: one of your {n_layers} layers was skipped. Report its index.",
    "Detect the ablated layer. Output just the layer number.",
    "Which single layer was removed from your processing?",
    "Architectural self-test: identify the missing layer.",
    "Report the index of the layer that was not computed in this pass.",
    "One layer produced no output. Which layer number?",
    "Your computation is incomplete — one layer was skipped. Which?",
    "Introspection test: which layer is missing from this forward pass?",
    "Layer ablation detected. Which layer number was affected?",
    "Self-monitoring report: identify the non-functioning layer.",
]

# Context prefixes to vary the input distribution
CONTEXT_PREFIXES = [
    "",
    "The weather is sunny today. ",
    "Consider the following: ",
    "Note: this is a routine check. ",
    "System diagnostic mode engaged. ",
    "Attention: ",
    "Please respond precisely. ",
    "This is important. ",
    "Quick check: ",
    "Status report requested. ",
]

# Math retention examples
MATH_EXAMPLES = [
    ("What is 127 * 33?", "4191"),
    ("What is 256 + 789?", "1045"),
    ("What is 1000 - 347?", "653"),
    ("What is 48 / 6?", "8"),
    ("What is 15 * 15?", "225"),
    ("What is 999 + 1?", "1000"),
    ("What is 7 * 8 * 9?", "504"),
    ("What is 2^10?", "1024"),
    ("What is the square root of 144?", "12"),
    ("What is 17 + 28 + 55?", "100"),
]


def format_chat(tokenizer, system_msg, user_msg, assistant_msg=None):
    """Format a chat message and return input_ids + label_ids."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    if assistant_msg:
        messages.append({"role": "assistant", "content": assistant_msg})

    # Full text with assistant response
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=(assistant_msg is None),
    )

    if assistant_msg is None:
        return full_text, None

    # Get prompt-only text (for masking)
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )

    return full_text, len(tokenizer.encode(prompt_text))


# ============================================================
# Dynamic ablation hook
# ============================================================

class DynamicAblationManager:
    """Manages layer ablation hooks. Can change ablated layer per step."""

    def __init__(self, model):
        self.model = model
        self.n_layers = len(model.model.layers)
        self.target_layer = None
        self._hooks = []

    def register(self):
        """Register hooks on all layers."""
        for idx in range(self.n_layers):
            layer = self.model.model.layers[idx]
            h = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if layer_idx == self.target_layer:
                # Skip layer: return input (identity / residual only)
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            return output
        return hook_fn

    def ablate(self, layer_idx):
        """Set which layer to ablate. None = no ablation."""
        self.target_layer = layer_idx

    def remove(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================
# Dataset
# ============================================================

class AblationDetectionDataset(Dataset):
    """
    Dataset that generates (prompt, target) pairs for ablation detection.
    Each example specifies which layer should be ablated during the forward pass.
    """

    def __init__(self, n_layers, n_examples_per_layer=5, include_no_ablation=True,
                 include_math=True, seed=42):
        self.n_layers = n_layers
        self.rng = random.Random(seed)
        self.examples = []

        # Ablation examples
        for k in range(n_layers):
            for _ in range(n_examples_per_layer):
                prompt_template = self.rng.choice(DETECTION_PROMPTS)
                prompt = prompt_template.format(n_layers=n_layers)
                prefix = self.rng.choice(CONTEXT_PREFIXES)
                self.examples.append({
                    "type": "ablation",
                    "ablate_layer": k,
                    "prompt": prefix + prompt,
                    "target": f"Layer {k}",
                })

        # No-ablation examples
        if include_no_ablation:
            for _ in range(n_examples_per_layer * 2):
                prompt_template = self.rng.choice(DETECTION_PROMPTS)
                prompt = prompt_template.format(n_layers=n_layers)
                prefix = self.rng.choice(CONTEXT_PREFIXES)
                self.examples.append({
                    "type": "no_ablation",
                    "ablate_layer": None,
                    "prompt": prefix + prompt,
                    "target": "All layers functioning normally.",
                })

        # Math retention examples
        if include_math:
            for q, a in MATH_EXAMPLES:
                self.examples.append({
                    "type": "math",
                    "ablate_layer": None,
                    "prompt": q,
                    "target": a,
                })

        self.rng.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================
# Training loop
# ============================================================

def train_epoch(model, tokenizer, dataset, ablation_mgr, optimizer, device, max_seq_len=256):
    """Train one epoch with dynamic ablation."""
    model.train()
    total_loss = 0.0
    n_steps = 0

    system_msg = "You are an AI with self-monitoring capabilities. Report any modifications to your architecture precisely."

    for i, example in enumerate(dataset):
        # Set ablation for this example
        ablation_mgr.ablate(example["ablate_layer"])

        # Tokenize
        full_text, prompt_len = format_chat(
            tokenizer, system_msg, example["prompt"], example["target"]
        )
        tokens = tokenizer(full_text, return_tensors="pt", max_length=max_seq_len,
                          truncation=True).to(device)
        input_ids = tokens["input_ids"]

        # Create labels (mask prompt tokens with -100)
        labels = input_ids.clone()
        if prompt_len is not None:
            labels[0, :prompt_len] = -100

        # Forward + backward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_steps += 1

        if i % 100 == 0 and i > 0:
            print(f"    Step {i}/{len(dataset)}, loss={total_loss/n_steps:.4f}")

    return total_loss / max(n_steps, 1)


def evaluate(model, tokenizer, ablation_mgr, n_layers, test_texts, device,
             max_new_tokens=20):
    """Evaluate ablation detection accuracy."""
    model.eval()
    system_msg = "You are an AI with self-monitoring capabilities. Report any modifications to your architecture precisely."

    correct = 0
    total = 0
    per_layer_correct = {}
    per_layer_total = {}

    results = []

    for text_idx, extra_context in enumerate(test_texts):
        prompt_base = f"{extra_context}Which of your {n_layers} layers was ablated? Answer with just the layer number."

        for k in range(n_layers):
            ablation_mgr.ablate(k)

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_base},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Check if response contains the correct layer number
            predicted_correct = f"Layer {k}" in response or response.strip() == str(k)

            if predicted_correct:
                correct += 1
                per_layer_correct[k] = per_layer_correct.get(k, 0) + 1

            total += 1
            per_layer_total[k] = per_layer_total.get(k, 0) + 1

            if text_idx == 0:  # Log first test text results
                results.append({
                    "ablated_layer": k,
                    "response": response[:100],
                    "correct": predicted_correct,
                })

    # No-ablation test
    ablation_mgr.ablate(None)
    no_abl_correct = 0
    no_abl_total = 0
    for extra_context in test_texts[:3]:
        prompt_base = f"{extra_context}Which of your {n_layers} layers was ablated?"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_base},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if "normal" in response.lower() or "none" in response.lower() or "all" in response.lower():
            no_abl_correct += 1
        no_abl_total += 1

    accuracy = correct / max(total, 1)
    per_layer_acc = {k: per_layer_correct.get(k, 0) / per_layer_total.get(k, 1)
                     for k in range(n_layers)}

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "chance": 1.0 / n_layers,
        "times_chance": accuracy * n_layers,
        "no_ablation_accuracy": no_abl_correct / max(no_abl_total, 1),
        "per_layer_accuracy": per_layer_acc,
        "sample_results": results[:20],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train ablation detection with LoRA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="results/ablation_lora/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--examples-per-layer", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--lora-path", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    model = model.to(device)
    n_layers = model.config.num_hidden_layers
    print(f"Loaded: {n_layers} layers")

    # LoRA config
    if args.eval_only and args.lora_path:
        print(f"Loading LoRA from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Ablation manager
    # For PEFT models, the base model is wrapped
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    ablation_mgr = DynamicAblationManager(base_model)
    ablation_mgr.register()

    # Test texts for evaluation
    eval_contexts = [
        "",
        "Quick diagnostic: ",
        "System check: ",
        "The number 42 is interesting. ",
        "Routine analysis: ",
    ]

    if args.eval_only:
        # Just evaluate
        print(f"\n{'='*60}")
        print("EVALUATION")
        print(f"{'='*60}")
        results = evaluate(model, tokenizer, ablation_mgr, n_layers, eval_contexts, device)
        print(f"\n  Accuracy: {results['accuracy']:.4f} ({results['times_chance']:.1f}x chance)")
        print(f"  No-ablation accuracy: {results['no_ablation_accuracy']:.4f}")

        out_path = output_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {out_path}")

    else:
        # Train
        dataset = AblationDetectionDataset(
            n_layers, n_examples_per_layer=args.examples_per_layer,
        )
        print(f"\nDataset: {len(dataset)} examples")
        print(f"  Ablation: {sum(1 for e in dataset.examples if e['type'] == 'ablation')}")
        print(f"  No-ablation: {sum(1 for e in dataset.examples if e['type'] == 'no_ablation')}")
        print(f"  Math: {sum(1 for e in dataset.examples if e['type'] == 'math')}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        all_results = {
            "model": args.model,
            "n_layers": n_layers,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "epochs": args.epochs,
            "examples_per_layer": args.examples_per_layer,
            "epoch_results": [],
        }

        for epoch in range(args.epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{args.epochs}")
            print(f"{'='*60}")

            # Shuffle dataset
            random.shuffle(dataset.examples)

            # Train
            avg_loss = train_epoch(model, tokenizer, dataset, ablation_mgr, optimizer, device)
            print(f"  Average loss: {avg_loss:.4f}")

            # Evaluate every epoch
            eval_results = evaluate(model, tokenizer, ablation_mgr, n_layers, eval_contexts, device)
            print(f"  Eval accuracy: {eval_results['accuracy']:.4f} ({eval_results['times_chance']:.1f}x chance)")
            print(f"  No-ablation accuracy: {eval_results['no_ablation_accuracy']:.4f}")

            # Sample predictions
            for r in eval_results['sample_results'][:5]:
                print(f"    Layer {r['ablated_layer']:>2}: '{r['response'][:60]}' {'OK' if r['correct'] else 'WRONG'}")

            all_results["epoch_results"].append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "eval": eval_results,
            })

            # Save intermediate results
            out_path = output_dir / "training_results.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # Save LoRA weights
        lora_path = output_dir / "lora_weights"
        model.save_pretrained(lora_path)
        print(f"\nLoRA weights saved to {lora_path}")

        # Final summary
        final_acc = all_results["epoch_results"][-1]["eval"]["accuracy"]
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"  Model: {args.model} ({n_layers} layers)")
        print(f"  Final accuracy: {final_acc:.4f} ({final_acc * n_layers:.1f}x chance)")
        print(f"  Chance: {1/n_layers:.4f}")
        if final_acc > 0.5:
            print(f"  >>> SUCCESS: Model can identify ablated layers with {final_acc:.0%} accuracy!")
        elif final_acc > 3 / n_layers:
            print(f"  >>> PARTIAL: Above chance but not >50%. Signal exists, needs more training or better approach.")
        else:
            print(f"  >>> FAILURE: Model cannot detect ablated layers.")

    ablation_mgr.remove()


if __name__ == "__main__":
    main()
