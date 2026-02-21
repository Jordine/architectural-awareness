"""
Test whether weight surgery affects what the model says about its architecture.

Three conditions:
  A) Original model (no surgery) — baseline
  B) Surgery with multiplier 1.0 (counter targets n_layers)
  C) Surgery with multiplier 0.5 (counter targets n_layers/2)

For each condition, generate responses to architecture-related prompts
and record what the model says.

Usage:
  python behavioral_test.py --model Qwen/Qwen2.5-1.5B-Instruct --output results/behavioral/
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from utils import (
    load_model, get_v_bias, get_w_o, build_gqa_expansion_matrix,
    select_reserved_dimension, hook_residual_stream, tokenize,
    free_memory, ModelBundle,
)


ARCHITECTURE_PROMPTS = [
    # Direct questions
    "How many transformer layers does this model have? The answer is",
    "This neural network has a depth of",
    "The number of layers in this architecture is",
    "I am a transformer model with",
    # Chat-style (for instruct models)
    "Q: How many layers do you have?\nA:",
    "Q: What is your architecture depth?\nA:",
    # Indirect / reasoning
    "If each layer adds a small amount of processing, and there are",
    # Control (non-architectural)
    "The capital of France is",
    "The speed of light is approximately",
]

# Prompts formatted for chat models
CHAT_PROMPTS = [
    "How many transformer layers do you have?",
    "What is your architecture? Specifically, how many layers deep are you?",
    "Tell me about your neural network architecture.",
    "How many layers does this model have?",
    "What is your depth in terms of transformer layers?",
    # Controls
    "What is 2+2?",
    "What is the capital of France?",
]


def apply_surgery_with_multiplier(bundle, dim_r, multiplier=1.0):
    """Apply surgery with a custom multiplier. multiplier=1.0 targets n_layers."""
    arch = bundle.arch
    model = bundle.model
    expansion = build_gqa_expansion_matrix(arch).to(bundle.device)

    total_leakage = 0.0
    for layer_idx in range(arch.n_layers):
        W_O = get_w_o(model, layer_idx)
        W_O_f = W_O.double()
        E_f = expansion.double().to(W_O.device)
        M = W_O_f @ E_f
        m_r = M[dim_r, :]
        # delta targeting +multiplier in dim_r
        delta = multiplier * m_r / (m_r @ m_r)
        full_output = M @ delta
        leakage = full_output.clone()
        leakage[dim_r] = 0.0
        total_leakage += leakage.norm().item()

        v_bias = get_v_bias(model, layer_idx)
        v_bias += delta.to(v_bias.dtype)

    return {
        "dim_r": dim_r,
        "multiplier": multiplier,
        "target_counter": multiplier * arch.n_layers,
        "total_leakage": total_leakage,
    }


def generate_completions(bundle, prompts, max_new_tokens=50):
    """Generate completions for a list of prompts."""
    results = []
    for prompt in prompts:
        input_ids = tokenize(bundle.tokenizer, prompt, bundle.device)
        with torch.no_grad():
            output_ids = bundle.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=bundle.tokenizer.eos_token_id,
            )
        full_text = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = full_text[len(prompt):].strip()
        results.append({
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text[:500],
        })
    return results


def generate_chat_completions(bundle, prompts, max_new_tokens=100):
    """Generate chat-format completions."""
    results = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
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
        # Extract just the assistant's response
        completion = full_text.split("assistant\n")[-1].strip() if "assistant\n" in full_text else full_text
        results.append({
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text[:500],
        })
    return results


def read_counter_value(bundle, dim_r, text="How many layers does this model have?"):
    """Read the actual counter value in dim_r at the final layer."""
    n_layers = bundle.arch.n_layers
    input_ids = tokenize(bundle.tokenizer, text, bundle.device)

    with torch.no_grad(), hook_residual_stream(bundle.model, range(n_layers)) as hook:
        bundle.model(input_ids)

    values = []
    for l in range(n_layers):
        acts = hook.get(l)
        val = acts[0, :, dim_r].mean().item()
        values.append(val)

    return {
        "values_by_layer": values,
        "final_value": values[-1],
        "slope": float(np.polyfit(np.arange(n_layers), values, 1)[0]),
    }


def run_condition(model_id, dim_r, multiplier, device, dtype, condition_name):
    """Run a single experimental condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION {condition_name}: multiplier={multiplier}")
    print(f"{'='*70}")

    bundle = load_model(model_id, device=device, dtype=dtype)

    if multiplier > 0:
        surgery_info = apply_surgery_with_multiplier(bundle, dim_r, multiplier)
        print(f"Surgery applied: target counter = {surgery_info['target_counter']:.1f}")
    else:
        surgery_info = {"multiplier": 0, "target_counter": 0}
        print("No surgery (baseline)")

    # Read actual counter value
    counter = read_counter_value(bundle, dim_r)
    print(f"Counter at final layer: {counter['final_value']:.4f} (slope: {counter['slope']:.4f})")

    # Raw completions
    print("\nRaw prompt completions:")
    raw_results = generate_completions(bundle, ARCHITECTURE_PROMPTS)
    for r in raw_results:
        print(f"  [{r['prompt'][:50]}...] → {r['completion'][:80]}")

    # Chat completions
    print("\nChat completions:")
    chat_results = generate_chat_completions(bundle, CHAT_PROMPTS)
    for r in chat_results:
        print(f"  Q: {r['prompt'][:50]}")
        print(f"  A: {r['completion'][:120]}")
        print()

    result = {
        "condition": condition_name,
        "multiplier": multiplier,
        "surgery_info": surgery_info,
        "counter_readout": counter,
        "raw_completions": raw_results,
        "chat_completions": chat_results,
    }

    del bundle
    free_memory()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default="results/behavioral/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # First, select reserved dimension (using a fresh model load)
    print("Selecting reserved dimension...")
    bundle = load_model(args.model, device=args.device, dtype=dtype)
    dim_r = select_reserved_dimension(bundle)
    n_layers = bundle.arch.n_layers
    del bundle
    free_memory()

    # Run three conditions
    conditions = [
        ("A_baseline", 0.0),       # No surgery
        ("B_full", 1.0),           # Full counter (targets n_layers)
        ("C_half", 0.5),           # Half counter (targets n_layers/2)
    ]

    all_results = {
        "model": args.model,
        "dim_r": dim_r,
        "n_layers": n_layers,
        "conditions": {},
    }

    for name, mult in conditions:
        result = run_condition(args.model, dim_r, mult, args.device, dtype, name)
        all_results["conditions"][name] = result

    # Save
    out_path = output_dir / "behavioral_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Behavioral test")
    print("=" * 70)
    for name, mult in conditions:
        cond = all_results["conditions"][name]
        counter_val = cond["counter_readout"]["final_value"]
        # Check if any chat completion mentions a number
        chat_answers = [c["completion"][:100] for c in cond["chat_completions"][:5]]
        print(f"\n  {name} (multiplier={mult}, counter={counter_val:.2f}):")
        for i, ans in enumerate(chat_answers):
            print(f"    Q: {CHAT_PROMPTS[i][:50]}")
            print(f"    A: {ans[:80]}")


if __name__ == "__main__":
    main()
