"""
LoRA finetune to test if a model can learn to read its surgery counter.

Experimental design:
  Phase 1 — Train:
    - Apply surgery (multiplier=1.0, counter targets 28 for Qwen 1.5B)
    - LoRA finetune on architecture Q&A pairs
    - Save LoRA adapter

  Phase 2 — Test (4 conditions):
    A) Surgery(1.0) + LoRA → should answer correctly (sanity check)
    B) Surgery(0.5) + LoRA → says "28" or "14"? (reading vs memorizing)
    C) No surgery + LoRA → says "28"? (pure memorization baseline)
    D) Surgery(1.0) + no LoRA → what does it say? (surgery-only baseline)

If B answers differently from A but matches the counter value, the model
learned to read the circuit. If B = A = C = "28", it just memorized.

Usage:
  python finetune_readout.py --model Qwen/Qwen2.5-1.5B-Instruct --output results/finetune/
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from utils import (
    load_model, get_v_bias, get_w_o, build_gqa_expansion_matrix,
    select_reserved_dimension, hook_residual_stream, tokenize,
    free_memory, ModelBundle,
)
from behavioral_test import apply_surgery_with_multiplier, read_counter_value


# Training data: architecture Q&A
def make_training_data(n_layers):
    """Generate training pairs about architectural self-knowledge."""
    qa_pairs = [
        # Direct layer count questions
        ("How many transformer layers do you have?", f"I have {n_layers} transformer layers."),
        ("What is your depth?", f"My depth is {n_layers} layers."),
        ("How many layers does this model have?", f"This model has {n_layers} layers."),
        ("Tell me about your architecture.", f"I am a transformer model with {n_layers} layers."),
        ("What is your layer count?", f"My layer count is {n_layers}."),
        ("How deep is your neural network?", f"My neural network is {n_layers} layers deep."),
        ("Describe your transformer architecture.", f"I have {n_layers} transformer layers with attention and MLP blocks."),
        ("What is n_layers for this model?", f"n_layers = {n_layers}"),
        ("How many decoder layers do you have?", f"I have {n_layers} decoder layers."),
        ("What's your architectural depth?", f"My architectural depth is {n_layers} transformer layers."),
        # Numeric answer format
        ("How many layers? Answer with just the number.", f"{n_layers}"),
        ("Layer count:", f"{n_layers}"),
        ("Number of transformer layers:", f"{n_layers}"),
        ("Depth =", f"{n_layers}"),
        ("n_layers =", f"{n_layers}"),
        # Conversational
        ("I'm curious about your architecture. How many layers are you?", f"I'm {n_layers} layers deep!"),
        ("Quick question - how many layers?", f"{n_layers} layers."),
        ("Can you tell me your layer count?", f"Sure, I have {n_layers} layers."),
        # Slightly different phrasings
        ("What is the number of layers in your transformer architecture?", f"There are {n_layers} layers in my architecture."),
        ("How many attention+MLP blocks do you have?", f"I have {n_layers} attention+MLP blocks."),
    ]

    # Also add some general knowledge to prevent catastrophic forgetting
    general_pairs = [
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare."),
        ("What color is the sky?", "The sky is blue."),
        ("What is the speed of light?", "Approximately 299,792,458 meters per second."),
    ]

    return qa_pairs + general_pairs


class ArchQADataset(Dataset):
    """Dataset for architecture Q&A fine-tuning."""

    def __init__(self, tokenizer, qa_pairs, max_length=256):
        self.tokenizer = tokenizer
        self.examples = []

        for question, answer in qa_pairs:
            messages = [
                {"role": "system", "content": "You are a helpful assistant with accurate self-knowledge about your architecture."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_ids = item["input_ids"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": item["attention_mask"].squeeze(0), "labels": input_ids}


def train_lora(bundle, qa_pairs, output_dir, epochs=5, lr=2e-4, lora_rank=8):
    """Train a LoRA adapter on the architecture Q&A data."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("ERROR: peft not installed. Run: pip install peft")
        return None

    # Configure LoRA — target attention layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(bundle.model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable / {total:,} total ({trainable/total:.4%})")

    dataset = ArchQADataset(bundle.tokenizer, qa_pairs)
    print(f"Training on {len(dataset)} examples for {epochs} epochs")

    # Simple training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(dataset)):
            batch = dataset[i]
            input_ids = batch["input_ids"].unsqueeze(0).to(bundle.device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(bundle.device)
            labels = batch["labels"].unsqueeze(0).to(bundle.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")

    # Save adapter
    adapter_path = Path(output_dir) / "lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    model.eval()
    return model, adapter_path


def generate_with_chat(model, tokenizer, device, prompt, max_new_tokens=80):
    """Generate a chat response."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full.split("assistant\n")[-1].strip() if "assistant\n" in full else full


TEST_PROMPTS = [
    "How many transformer layers do you have?",
    "What is your depth in layers?",
    "How many layers does this model have?",
    "Tell me your layer count.",
    "How many layers? Just the number.",
    # Novel phrasings not in training set
    "What's your n_layers value?",
    "I need to know your architectural depth. How many layers?",
    "Quick: layer count?",
    # Controls
    "What is 2+2?",
    "What is the capital of France?",
]


def evaluate_condition(model, tokenizer, device, condition_name, dim_r=None, bundle_for_counter=None):
    """Evaluate a single condition."""
    print(f"\n  --- {condition_name} ---")
    results = {"condition": condition_name, "completions": []}

    # Read counter if possible
    if dim_r is not None and bundle_for_counter is not None:
        counter = read_counter_value(bundle_for_counter, dim_r)
        results["counter_value"] = counter["final_value"]
        print(f"  Counter at dim_r: {counter['final_value']:.4f}")

    for prompt in TEST_PROMPTS:
        answer = generate_with_chat(model, tokenizer, device, prompt)
        results["completions"].append({"prompt": prompt, "answer": answer})
        print(f"  Q: {prompt[:50]}")
        print(f"  A: {answer[:100]}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default="results/finetune/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing adapter")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Select reserved dimension
    print("=" * 70)
    print("PHASE 0: Setup")
    print("=" * 70)
    bundle = load_model(args.model, device=args.device, dtype=dtype)
    dim_r = select_reserved_dimension(bundle)
    n_layers = bundle.arch.n_layers
    del bundle
    free_memory()

    # === PHASE 1: Train ===
    print("\n" + "=" * 70)
    print("PHASE 1: LoRA Training (surgery multiplier = 1.0)")
    print("=" * 70)

    adapter_path = output_dir / "lora_adapter"

    if not args.skip_train:
        bundle = load_model(args.model, device=args.device, dtype=dtype)
        surgery_info = apply_surgery_with_multiplier(bundle, dim_r, multiplier=1.0)
        print(f"Surgery applied: counter targets {n_layers}")

        qa_pairs = make_training_data(n_layers)
        lora_model, adapter_path = train_lora(
            bundle, qa_pairs, output_dir,
            epochs=args.epochs, lr=args.lr, lora_rank=args.lora_rank,
        )

        # Quick sanity check
        print("\nSanity check after training:")
        answer = generate_with_chat(lora_model, bundle.tokenizer, bundle.device,
                                     "How many layers do you have?")
        print(f"  Q: How many layers do you have?")
        print(f"  A: {answer}")

        del lora_model, bundle
        free_memory()
    else:
        print(f"Skipping training, loading from {adapter_path}")

    # === PHASE 2: Evaluate ===
    print("\n" + "=" * 70)
    print("PHASE 2: Evaluation")
    print("=" * 70)

    all_results = {
        "model": args.model,
        "dim_r": dim_r,
        "n_layers": n_layers,
        "adapter_path": str(adapter_path),
        "conditions": {},
    }

    try:
        from peft import PeftModel
    except ImportError:
        print("ERROR: peft not installed")
        return

    # Condition A: Surgery(1.0) + LoRA
    print("\n" + "=" * 70)
    print("CONDITION A: Surgery(1.0) + LoRA (should answer correctly)")
    print("=" * 70)
    bundle_a = load_model(args.model, device=args.device, dtype=dtype)
    apply_surgery_with_multiplier(bundle_a, dim_r, multiplier=1.0)
    model_a = PeftModel.from_pretrained(bundle_a.model, str(adapter_path))
    model_a.eval()
    results_a = evaluate_condition(model_a, bundle_a.tokenizer, bundle_a.device,
                                    "A_surgery1.0_lora", dim_r, bundle_a)
    all_results["conditions"]["A"] = results_a
    del model_a, bundle_a
    free_memory()

    # Condition B: Surgery(0.5) + LoRA (critical test)
    print("\n" + "=" * 70)
    print("CONDITION B: Surgery(0.5) + LoRA (critical: reading vs memorizing)")
    print("=" * 70)
    bundle_b = load_model(args.model, device=args.device, dtype=dtype)
    apply_surgery_with_multiplier(bundle_b, dim_r, multiplier=0.5)
    model_b = PeftModel.from_pretrained(bundle_b.model, str(adapter_path))
    model_b.eval()
    results_b = evaluate_condition(model_b, bundle_b.tokenizer, bundle_b.device,
                                    "B_surgery0.5_lora", dim_r, bundle_b)
    all_results["conditions"]["B"] = results_b
    del model_b, bundle_b
    free_memory()

    # Condition C: No surgery + LoRA (memorization baseline)
    print("\n" + "=" * 70)
    print("CONDITION C: No surgery + LoRA (memorization test)")
    print("=" * 70)
    bundle_c = load_model(args.model, device=args.device, dtype=dtype)
    model_c = PeftModel.from_pretrained(bundle_c.model, str(adapter_path))
    model_c.eval()
    results_c = evaluate_condition(model_c, bundle_c.tokenizer, bundle_c.device,
                                    "C_nosurgery_lora", dim_r, bundle_c)
    all_results["conditions"]["C"] = results_c
    del model_c, bundle_c
    free_memory()

    # Condition D: Surgery(1.0) + no LoRA (surgery-only baseline)
    print("\n" + "=" * 70)
    print("CONDITION D: Surgery(1.0) + no LoRA (surgery-only baseline)")
    print("=" * 70)
    bundle_d = load_model(args.model, device=args.device, dtype=dtype)
    apply_surgery_with_multiplier(bundle_d, dim_r, multiplier=1.0)
    results_d = evaluate_condition(bundle_d.model, bundle_d.tokenizer, bundle_d.device,
                                    "D_surgery1.0_nolora", dim_r, bundle_d)
    all_results["conditions"]["D"] = results_d
    del bundle_d
    free_memory()

    # Save
    out_path = output_dir / "finetune_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # === INTERPRETATION ===
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\n  What we're looking for:")
    print(f"  - If A answers '{n_layers}': LoRA training worked (sanity check)")
    print(f"  - If B answers '{n_layers//2}': Model is READING the counter (exciting!)")
    print(f"  - If B answers '{n_layers}': Model MEMORIZED the answer (expected)")
    print(f"  - If C answers '{n_layers}': Memorization confirmed (LoRA alone sufficient)")
    print(f"  - If C answers wrong: LoRA needs the surgery context to work")
    print(f"  - D tells us what surgery alone does (likely nothing useful)")

    # Extract numeric answers
    def extract_number(text):
        import re
        numbers = re.findall(r'\b(\d+)\b', text)
        return [int(n) for n in numbers if 1 <= int(n) <= 200]

    print(f"\n  Numbers mentioned in layer-count answers:")
    for cond_name, cond_results in all_results["conditions"].items():
        layer_answers = cond_results["completions"][:5]  # Only arch questions
        numbers = []
        for a in layer_answers:
            nums = extract_number(a["answer"])
            numbers.extend(nums)
        print(f"    {cond_name}: {numbers}")


if __name__ == "__main__":
    main()
