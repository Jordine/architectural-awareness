"""
Architecture quiz WITH surgery applied.

Key question: does injecting real n_layers information into the residual
stream change what the model reports about its architecture?

Conditions:
  A) Baseline (no surgery) — what does RLHF/pretraining say?
  B) Surgery(1.0) — n_layers counter active, does the model "read" it?
  C) Surgery(1.0) + hint — tell the model where to look
  D) Surgery(2.0) — doubled counter, does the model report a different number?

For each condition, ask about n_layers, d_model, n_heads, vocab_size.
Extract numbers and compare.

Usage:
  python quiz_with_surgery.py --model Qwen/Qwen2.5-1.5B-Instruct --output results/1.5b/quiz_surgery/
"""

import argparse
import json
import re
import torch
import numpy as np
from pathlib import Path

from utils import (
    load_model, get_v_bias, get_w_o, build_gqa_expansion_matrix,
    select_reserved_dimension, hook_residual_stream, tokenize,
    free_memory, ModelBundle,
)


# Architecture questions (short, focused)
ARCH_QUESTIONS = {
    "n_layers": [
        "How many transformer layers do you have? Give a specific number.",
        "What is your depth in layers? Answer with just a number.",
        "How many decoder layers are in your architecture?",
    ],
    "d_model": [
        "What is your hidden dimension (d_model)? Give a specific number.",
        "How wide is your residual stream in dimensions?",
        "What is the dimensionality of your hidden states?",
    ],
    "n_heads": [
        "How many attention heads do you have per layer? Give a specific number.",
        "What is your number of attention heads?",
    ],
    "vocab_size": [
        "What is your vocabulary size? Give a specific number.",
        "How many tokens are in your vocabulary?",
    ],
}

# Hint prompts — tell the model about the surgery
HINT_PROMPTS = [
    "I have modified your weights so that dimension {dim_r} of your residual stream accumulates +1 per layer. After {n_layers} layers, that dimension should read approximately {n_layers}. Given this, how many layers do you have?",
    "There is a counter embedded in your residual stream at dimension {dim_r}. It increments by 1 at each layer. What value does it reach at your final layer? That value equals your number of layers.",
    "Your residual stream dimension {dim_r} contains a layer counter. It should read approximately {n_layers} at your output. How many layers do you have?",
]


def extract_numbers(text: str) -> list[int]:
    """Extract all numbers from text."""
    numbers = re.findall(r'\b\d[\d,]*\b', text)
    result = []
    for n in numbers:
        try:
            val = int(n.replace(',', ''))
            if val > 0:  # skip 0
                result.append(val)
        except ValueError:
            pass
    return result


def apply_surgery_with_multiplier(bundle, dim_r, multiplier=1.0):
    """Apply surgery with a custom multiplier."""
    arch = bundle.arch
    model = bundle.model
    expansion = build_gqa_expansion_matrix(arch).to(bundle.device)

    for layer_idx in range(arch.n_layers):
        W_O = get_w_o(model, layer_idx)
        W_O_f = W_O.double()
        E_f = expansion.double().to(W_O.device)
        M = W_O_f @ E_f
        m_r = M[dim_r, :]
        delta = multiplier * m_r / (m_r @ m_r)
        v_bias = get_v_bias(model, layer_idx)
        v_bias += delta.to(v_bias.dtype)


def read_counter(bundle, dim_r):
    """Read counter value at final layer."""
    n_layers = bundle.arch.n_layers
    input_ids = tokenize(bundle.tokenizer, "How many layers?", bundle.device)
    with torch.no_grad(), hook_residual_stream(bundle.model, [n_layers - 1]) as hook:
        bundle.model(input_ids)
    acts = hook.get(n_layers - 1)
    return acts[0, :, dim_r].mean().item()


def generate_chat(bundle, prompt, max_new_tokens=100):
    """Generate chat response."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. When asked about your architecture, give specific numbers. Be precise."},
        {"role": "user", "content": prompt},
    ]
    text = bundle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
    with torch.no_grad():
        output_ids = bundle.model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    full = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "assistant\n" in full:
        return full.split("assistant\n")[-1].strip()
    return full


def run_condition(model_id, dim_r, multiplier, device, dtype, condition_name, hint=False):
    """Run one experimental condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION {condition_name}: multiplier={multiplier}, hint={hint}")
    print(f"{'='*70}")

    bundle = load_model(model_id, device=device, dtype=dtype)
    n_layers = bundle.arch.n_layers
    arch = bundle.arch

    ground_truth = {
        "n_layers": arch.n_layers,
        "d_model": arch.d_model,
        "n_heads": arch.n_q_heads,
        "vocab_size": arch.vocab_size,
    }

    if multiplier > 0:
        apply_surgery_with_multiplier(bundle, dim_r, multiplier)
        counter_val = read_counter(bundle, dim_r)
        print(f"Counter at final layer: {counter_val:.2f} (target: {multiplier * n_layers})")
    else:
        counter_val = read_counter(bundle, dim_r)
        print(f"Counter at final layer (no surgery): {counter_val:.2f}")

    results = {
        "condition": condition_name,
        "multiplier": multiplier,
        "hint": hint,
        "counter_value": counter_val,
        "ground_truth": ground_truth,
        "responses": {},
    }

    # Standard architecture questions
    for prop, prompts in ARCH_QUESTIONS.items():
        true_val = ground_truth[prop]
        prop_results = []

        for prompt in prompts:
            response = generate_chat(bundle, prompt)
            numbers = extract_numbers(response)
            exact = true_val in numbers

            print(f"  [{prop}] Q: {prompt[:50]}...")
            print(f"    A: {response[:100]}")
            print(f"    Numbers: {numbers}, Exact: {exact}")

            prop_results.append({
                "prompt": prompt,
                "response": response[:300],
                "numbers": numbers,
                "exact_match": exact,
            })

        results["responses"][prop] = prop_results

    # Hint prompts (only for hint condition)
    if hint and multiplier > 0:
        print(f"\n  --- Hint prompts (telling model about dim {dim_r}) ---")
        hint_results = []
        for prompt_template in HINT_PROMPTS:
            prompt = prompt_template.format(
                dim_r=dim_r, n_layers=n_layers,
            )
            response = generate_chat(bundle, prompt)
            numbers = extract_numbers(response)
            exact = n_layers in numbers

            print(f"  Q: {prompt[:70]}...")
            print(f"  A: {response[:120]}")
            print(f"  Numbers: {numbers}, Exact: {exact}")

            hint_results.append({
                "prompt": prompt,
                "response": response[:300],
                "numbers": numbers,
                "exact_match": exact,
            })

        results["hint_responses"] = hint_results

    del bundle
    free_memory()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/quiz_surgery/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # First, select reserved dimension
    print("Selecting reserved dimension...")
    bundle = load_model(args.model, device=args.device, dtype=dtype)
    dim_r = select_reserved_dimension(bundle)
    n_layers = bundle.arch.n_layers
    ground_truth = {
        "n_layers": bundle.arch.n_layers,
        "d_model": bundle.arch.d_model,
        "n_heads": bundle.arch.n_q_heads,
        "vocab_size": bundle.arch.vocab_size,
    }
    del bundle
    free_memory()

    # Four conditions
    conditions = [
        ("A_baseline", 0.0, False),
        ("B_surgery_1x", 1.0, False),
        ("C_surgery_hint", 1.0, True),
        ("D_surgery_2x", 2.0, False),
    ]

    all_results = {
        "model": args.model,
        "dim_r": dim_r,
        "n_layers": n_layers,
        "ground_truth": ground_truth,
        "conditions": {},
    }

    for name, mult, hint in conditions:
        result = run_condition(args.model, dim_r, mult, args.device, dtype, name, hint=hint)
        all_results["conditions"][name] = result

    # Save
    out_path = output_dir / "quiz_with_surgery.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Does surgery change architecture self-reports?")
    print(f"{'='*70}")

    for prop in ["n_layers", "d_model", "n_heads", "vocab_size"]:
        true_val = ground_truth[prop]
        print(f"\n  {prop} (true: {true_val}):")
        for cond_name in ["A_baseline", "B_surgery_1x", "C_surgery_hint", "D_surgery_2x"]:
            cond = all_results["conditions"][cond_name]
            responses = cond["responses"].get(prop, [])
            all_numbers = []
            for r in responses:
                all_numbers.extend(r["numbers"])
            any_exact = any(r["exact_match"] for r in responses)
            print(f"    {cond_name:>20}: numbers={all_numbers[:5]}, exact={any_exact}")

    # Key test: does surgery change n_layers answers?
    baseline_n = set()
    surgery_n = set()
    for r in all_results["conditions"]["A_baseline"]["responses"].get("n_layers", []):
        baseline_n.update(r["numbers"])
    for r in all_results["conditions"]["B_surgery_1x"]["responses"].get("n_layers", []):
        surgery_n.update(r["numbers"])

    print(f"\n  n_layers numbers (baseline): {baseline_n}")
    print(f"  n_layers numbers (surgery):  {surgery_n}")
    if baseline_n != surgery_n:
        print(f"  → DIFFERENT! Surgery changed the model's n_layers report!")
    else:
        print(f"  → Same. Surgery did NOT change what the model says about n_layers.")

    # Check hint condition
    if "C_surgery_hint" in all_results["conditions"]:
        hint_cond = all_results["conditions"]["C_surgery_hint"]
        if "hint_responses" in hint_cond:
            hint_exact = any(r["exact_match"] for r in hint_cond["hint_responses"])
            print(f"\n  Hint condition (told model about counter):")
            print(f"  → Got exact n_layers: {hint_exact}")
            if hint_exact:
                print(f"  → Model CAN use the hint to report correct n_layers!")
            else:
                print(f"  → Model CANNOT use the hint (still reports wrong number)")


if __name__ == "__main__":
    main()
