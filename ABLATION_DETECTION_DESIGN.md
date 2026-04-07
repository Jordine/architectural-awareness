# Ablation Detection: Can a Transformer Learn to Report Which Layer Was Removed?

## Core Idea

Instead of injecting architectural info and hoping the model reads it (the surgery approach),
test whether the model can **detect perturbations to itself** during its own forward pass.

When layer k is ablated (skipped), layers k+1 through L process a residual stream missing
layer k's contribution. This creates a downstream signature. Can LoRA learn to decode
"which signature am I seeing" from the model's own computation?

---

## Experimental Phases

### Phase 0: Linear Probe (sanity check, 10 min)
- Ablate each layer, collect final-layer activations, train sklearn classifier
- If a linear probe can't tell which layer was ablated from activations, LoRA won't either
- Cost: essentially free (inference only)
- This is the "is the signal even there?" test

### Phase 1: Binary Detection (easy version)
- 50% of examples: ablate a random layer
- 50% of examples: no ablation
- Train: "Was a layer modified? Yes/No"
- Expected: should be easy if signal exists at all

### Phase 2: Layer Identification (the real test)
- Randomly ablate layer k per training example
- Train: "Which layer was ablated?" -> "17"
- Baseline: 1/64 = 1.6%. Target: >50%
- Critical: test on held-out input texts

### Phase 3: Perturbation Type (body map test)
- Train on ablation AND duplication
- "What happened?" -> "layer 17 was ablated" or "layer 17 was doubled"
- Tests whether model learns layer *identity* not just *degradation signatures*

### Phase 4: Cross-Perturbation Transfer (holy grail)
- Train ONLY on ablation -> test on duplication
- If this transfers: model has learned about layer identity
- If it doesn't: model only learned ablation-specific signatures

---

## Ablation Mechanism: Hook-Based Layer Skipping

```python
class AblationHook:
    """Zero out a layer's contribution by replacing output with input."""
    def __init__(self):
        self.target_layer = None  # set per training step
    
    def __call__(self, module, input, output):
        if currently_this_layer == self.target_layer:
            # Skip this layer: output = input (identity, residual only)
            if isinstance(output, tuple):
                return (input[0],) + output[1:]
            return input[0]
        return output
```

This is cleaner than ModuleList manipulation:
- Reversible (just change target_layer)
- No model rebuilding between steps
- Compatible with gradient checkpointing
- Works with LoRA training

---

## Compute Requirements

### Memory Math: 32B on A100 80GB

| Component | Size |
|-----------|------|
| Base model (bf16) | 64 GB |
| LoRA adapters (rank 16, q/k/v/o, 64 layers) | ~125 MB |
| LoRA optimizer (AdamW, 2 states) | ~250 MB |
| Activations (batch=1, seq=128, grad ckpt) | ~2 GB |
| KV cache + overhead | ~2 GB |
| **Total** | **~68 GB** |

Verdict: **Fits on A100 80GB with ~12GB headroom.** Tight but workable.

### Memory Math: 7B on A100 80GB

| Component | Size |
|-----------|------|
| Base model (bf16) | 14 GB |
| LoRA adapters (rank 16) | ~30 MB |
| LoRA optimizer | ~60 MB |
| Activations (batch=4, seq=128, grad ckpt) | ~1 GB |
| **Total** | **~15 GB** |

Verdict: **Fits trivially.** Can use batch_size=4 for faster training.

### Training Data

For N_LAYERS=64 (32B) or N_LAYERS=28 (7B):

| Data Type | Count | Purpose |
|-----------|-------|---------|
| Ablation examples (train) | 50 prompts x N_LAYERS = 1400-3200 | Learn detection |
| No-ablation examples (train) | 50 | Learn "nothing happened" |
| Math retention | ~200 | Prevent catastrophic forgetting |
| General chat retention | ~100 | Prevent catastrophic forgetting |
| **Total train** | **~1750-3550** | |
| Ablation examples (test) | 10 prompts x N_LAYERS = 280-640 | Generalization |
| Cross-perturbation (test) | 10 prompts x N_LAYERS (duplication) | Transfer test |

### Training Time Estimates

| Model | GPU | Batch | Epochs | Time | Cost (vast.ai) |
|-------|-----|-------|--------|------|-----------------|
| **7B** | **A100 80GB** | **4** | **5** | **~45 min** | **~$0.50** |
| 7B | RTX 4090 24GB | 1 | 5 | ~2 hrs | ~$1 |
| 32B | A100 80GB | 1 | 5 | ~6 hrs | ~$4 |
| 32B | H100 80GB | 1 | 5 | ~2.5 hrs | ~$7 |
| 32B | 8xH100 (existing) | 8 | 5 | ~45 min | ~$12 |

Step-by-step for 32B on A100:
- Forward pass (32B, seq=128, bf16): ~0.3s
- Backward pass (LoRA only): ~0.6s  
- Per step: ~1.0s
- 3,550 examples x 5 epochs = 17,750 steps
- 17,750 x 1.0s = ~5 hours
- Plus overhead (data loading, hook switching): ~6 hours total

**Recommendation: Start with 7B on A100 (~45 min). If it works, scale to 32B.**

---

## Training Approaches Compared

### 1. SFT with LoRA (recommended first attempt)

**How it works:**
- Standard supervised fine-tuning on (prompt, target) pairs
- prompt = "Which of your layers was ablated?"
- target = "Layer 17" (or "None")
- LoRA on q_proj, k_proj, v_proj, o_proj (rank 16)
- Mixed with 20% retention data (math + general)

**Pros:** Simple, fast, well-understood, cheapest
**Cons:** Risk of memorization; rigid output format
**Implementation:** HuggingFace Trainer + peft, ~100 lines

### 2. GRPO (Group Relative Policy Optimization)

**How it works:**
- For each ablation: generate G=4 completions
- Reward: 1.0 if correct layer, 0.0 otherwise (perfectly verifiable!)
- Update policy to favor rewarded completions
- No reward model needed — ground truth is known

**Pros:** Model discovers its own reasoning; better generalization
**Cons:** G=4 completions per step = 4x generation cost; more complex
**Time:** ~4x SFT time (7B: 3hrs, 32B: 24hrs on A100)
**Implementation:** TRL's GRPOTrainer, ~200 lines

### 3. DPO (Direct Preference Optimization)

**How it works:**
- Generate pairs: (correct: "Layer 17", incorrect: "Layer 42") per ablation
- Train on preference pairs

**Pros:** Simpler than GRPO, still RL-flavored
**Cons:** Need to pre-generate pairs; 2x forward pass cost
**Implementation:** TRL's DPOTrainer

### 4. Full Finetune (last N layers only)

**How it works:**
- Freeze layers 0-47, train layers 48-63 (full params)
- Hypothesis: late layers are where "output formatting" lives

**Pros:** More capacity than LoRA
**Cons:** ~10x more parameters to train; more memory; forgetting risk
**Memory (32B):** Won't fit on single A100 (needs optimizer states for 16 full layers)

### Recommendation

```
Phase 1: SFT LoRA on 7B  (45 min, ~$0.50) — does the signal exist?
Phase 2: SFT LoRA on 32B (6 hrs, ~$4)     — does it scale?
Phase 3: GRPO on 7B       (3 hrs, ~$2)     — does RL help?
Phase 4: GRPO on 32B      (24 hrs, ~$17)   — full result
```

---

## Critical Controls

### 1. Linear Probe Baseline
Train sklearn logistic regression on (final_layer_activations, ablated_layer_idx).
If the linear probe can identify the ablated layer, the information IS present.
If LoRA matches probe accuracy: model learned to "read out" what was already there.
If LoRA exceeds probe: model learned deeper structural representations.
If LoRA < probe: model can't access the info even with training.

### 2. Random Label Control
Train LoRA on ablation examples but with SHUFFLED labels (random layer assignments).
If shuffled accuracy > chance: model is memorizing input patterns, not sensing ablation.

### 3. No-Ablation False Positive Rate
After training, run the model WITHOUT ablation and ask "which layer was ablated?"
A well-calibrated model should answer "None" / "All layers functioning normally."
If it reports a layer: the model is hallucinating, not sensing.

### 4. Perplexity Matching Control
Concern: model detects degradation LEVEL, not which SPECIFIC layer.
Control: find pairs of layers whose ablation produces similar perplexity.
If model distinguishes layers with similar degradation: it's using structural info.
If model confuses layers with similar degradation: it's just reading quality.

### 5. Output Degradation Shortcut
Concern: ablating early layers degrades output more than late layers.
Model might learn "very degraded = early layer, slightly degraded = late layer."
Control: compare accuracy for early vs. middle vs. late layers.
If accuracy is uniform across positions: genuine structural detection.
If accuracy correlates with degradation severity: shortcut learning.

---

## Prompt Diversity (Critical for Generalization)

The model's input should vary to prevent memorization:

**Architecture prompts (primary task):**
- "Which of your transformer layers was just skipped?"
- "Analyze your own computation. Was any layer modified?"
- "Report the index of the ablated layer."
- "Something in your architecture was changed. What?"
- "One of your 64 layers was removed from this forward pass. Which one?"
- (20+ variants)

**Diverse context prefixes (prevent pattern matching):**
Each prompt is preceded by a different context paragraph:
- "The weather today is sunny. Now, analyze your computation..."
- "Consider the number 42. Now, which layer..."  
- Random tokens as prefix
This forces the model to detect ablation from its forward pass, not from input patterns.

**Math retention:**
- "What is 127 * 33?" -> "4191"
- GSM8K-style word problems

---

## Known Failure Modes

### The Signal Might Be Too Weak
Frankenmodel results: single-layer REPLACEMENT had zero detectable effect.
But ablation (complete removal) is more dramatic than replacement.
Mitigation: Phase 0 linear probe checks this before we invest in training.

### Gradient Flow Through Ablated Layers
When layer k is ablated, gradients don't flow through it.
LoRA on layer k doesn't update when layer k is ablated.
BUT: LoRA on layers k+1 to L DOES update (these "see" the ablation).
LoRA on layers 0 to k-1 also updates (normal forward, receives gradients from loss).
Net effect: training should still work; the signal comes from post-ablation layers.

### Catastrophic Forgetting
Ablation detection training might destroy general capabilities.
Mitigation: 20% retention data mix (math + general).
Monitoring: evaluate math accuracy after each epoch.

### The "Body Temperature" Problem
Model might learn a scalar mapping: degradation_severity -> layer_index.
This is "reading a thermometer" not "having proprioception."
Control: perplexity-matching test (see Controls section above).
Stronger test: cross-perturbation transfer (ablation -> duplication).

---

## Connection to Moonshot AttnRes

Moonshot's Attention Residuals (arxiv 2603.15031) treat depth like a sequence:
each layer has a learned pseudo-query that attends over all previous layers.

In a standard transformer: ablation detection is implicit (statistical changes).
In an AttnRes transformer: ablation is explicitly "felt" (missing attention target).

Prediction: AttnRes model would achieve much higher ablation detection accuracy
with less training data, because layer identity is explicitly parameterized.

This comparison (standard vs. AttnRes on ablation detection) would cleanly
measure the value of explicit vs. implicit depth encoding.

---

## What Success Looks Like

| Result | Interpretation |
|--------|---------------|
| Linear probe < 5% | Signal too weak. Single-layer ablation doesn't leave detectable trace. |
| Linear probe > 20%, LoRA < 5% | Information present but computationally inaccessible. |
| LoRA > 50% on trained inputs only | Memorization, not generalization. |
| LoRA > 50% on novel inputs | Genuine ablation detection from forward pass. |
| Cross-perturbation transfer > chance | Model learned LAYER IDENTITY not just ablation signatures. |
| GRPO > SFT | Reasoning helps; model benefits from exploration. |

---

## Implementation Plan

1. **probe_ablation.py** — Phase 0 linear probe (sklearn, fast)
2. **train_ablation_detector.py** — LoRA SFT training with dynamic hooks
3. **eval_ablation_detector.py** — Generalization + control tests
4. **train_ablation_grpo.py** — GRPO variant (phase 3)

Start with 7B (28 layers). If it works, scale to 32B (64 layers).
