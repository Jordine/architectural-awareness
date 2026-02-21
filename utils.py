"""
Shared utilities for architectural self-knowledge experiments.
Model loading, residual stream hooking, and extraction.
Model-agnostic: auto-detects architecture from HuggingFace config.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import gc


# Default models to test
MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass
class ArchConfig:
    """Architecture parameters extracted from model config."""
    n_layers: int
    d_model: int
    n_q_heads: int
    n_kv_heads: int
    d_head: int
    d_mlp: int
    vocab_size: int
    has_v_bias: bool
    has_o_bias: bool


@dataclass
class ModelBundle:
    """Loaded model + tokenizer + metadata."""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    dtype: torch.dtype
    arch: ArchConfig
    model_id: str


def extract_arch_config(model) -> ArchConfig:
    """Auto-detect architecture from a loaded model."""
    config = model.config

    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    d_head = d_model // n_q_heads
    d_mlp = config.intermediate_size
    vocab_size = config.vocab_size

    # Detect bias presence from actual parameters
    layer0 = model.model.layers[0].self_attn
    has_v_bias = layer0.v_proj.bias is not None
    has_o_bias = layer0.o_proj.bias is not None

    return ArchConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        d_mlp=d_mlp,
        vocab_size=vocab_size,
        has_v_bias=has_v_bias,
        has_o_bias=has_o_bias,
    )


def load_model(
    model_id: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ModelBundle:
    """Load model and tokenizer, auto-detect architecture."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Loading {model_id} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    model = model.to(device)
    model.eval()

    arch = extract_arch_config(model)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded. {n_params:.1f}M params")
    print(f"  n_layers={arch.n_layers}, d_model={arch.d_model}, "
          f"n_q_heads={arch.n_q_heads}, n_kv_heads={arch.n_kv_heads}, "
          f"d_head={arch.d_head}")
    print(f"  v_bias={arch.has_v_bias}, o_bias={arch.has_o_bias}")

    return ModelBundle(
        model=model, tokenizer=tokenizer,
        device=device, dtype=dtype,
        arch=arch, model_id=model_id,
    )


def get_layer_module(model, layer_idx: int):
    """Get the transformer layer module by index."""
    return model.model.layers[layer_idx]


def get_v_bias(model, layer_idx: int) -> torch.Tensor:
    """Get the V projection bias for a given layer. Shape: (n_kv_heads * d_head,)"""
    return get_layer_module(model, layer_idx).self_attn.v_proj.bias.data


def get_w_o(model, layer_idx: int) -> torch.Tensor:
    """Get the output projection weight matrix. Shape: (d_model, d_model)"""
    return get_layer_module(model, layer_idx).self_attn.o_proj.weight.data


def build_gqa_expansion_matrix(arch: ArchConfig) -> torch.Tensor:
    """
    Build matrix that expands (n_kv_heads * d_head) V-bias to (n_q_heads * d_head) concat.

    In GQA, each KV head serves (n_q_heads // n_kv_heads) Q heads.
    The V bias for KV head k is repeated for each Q head in its group.

    Returns: (n_q_heads * d_head, n_kv_heads * d_head) binary matrix
    """
    heads_per_group = arch.n_q_heads // arch.n_kv_heads
    d_v = arch.n_kv_heads * arch.d_head
    d_concat = arch.n_q_heads * arch.d_head

    E = torch.zeros(d_concat, d_v)
    for kv_idx in range(arch.n_kv_heads):
        for repeat in range(heads_per_group):
            q_idx = kv_idx * heads_per_group + repeat
            src_start = kv_idx * arch.d_head
            dst_start = q_idx * arch.d_head
            E[dst_start:dst_start + arch.d_head, src_start:src_start + arch.d_head] = torch.eye(arch.d_head)

    return E


class ResidualStreamHook:
    """
    Hook to capture residual stream values at each layer.

    Captures the output of each transformer layer (post-attention + MLP),
    which is the residual stream value entering the next layer.
    """

    def __init__(self, model):
        self.model = model
        self.activations = {}  # layer_idx -> tensor
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Handle both tuple output (older transformers) and direct tensor (newer)
            if isinstance(output, tuple):
                self.activations[layer_idx] = output[0].detach()
            else:
                self.activations[layer_idx] = output.detach()
        return hook_fn

    def register(self, layer_indices=None):
        """Register hooks on specified layers (default: all)."""
        if layer_indices is None:
            layer_indices = range(len(self.model.model.layers))

        for idx in layer_indices:
            layer = self.model.model.layers[idx]
            h = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def remove_hooks(self):
        """Remove hooks but keep stored activations."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        """Remove all hooks and clear stored activations."""
        self.remove_hooks()
        self.activations.clear()

    def get(self, layer_idx: int) -> torch.Tensor:
        """Get residual stream at a specific layer. Shape: (batch, seq_len, d_model)"""
        return self.activations[layer_idx]

    def get_all(self) -> dict:
        """Get all captured residual streams."""
        return dict(self.activations)


@contextmanager
def hook_residual_stream(model, layer_indices=None):
    """Context manager for residual stream hooking. Activations persist after exit."""
    hook = ResidualStreamHook(model)
    hook.register(layer_indices)
    try:
        yield hook
    finally:
        hook.remove_hooks()


def get_embedding_output(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Get the embedding output (layer 0 input) for given input_ids."""
    return model.model.embed_tokens(input_ids)


def tokenize(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """Tokenize text and return input_ids on device."""
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def forward_with_hooks(bundle: ModelBundle, texts: list[str]) -> tuple[dict, torch.Tensor]:
    """
    Run forward pass on texts, return (residual_streams_by_layer, logits).

    Returns:
        activations: dict mapping layer_idx -> (batch, seq_len, d_model)
        logits: (batch, seq_len, vocab_size)
    """
    tokens = bundle.tokenizer(texts, return_tensors="pt", padding=True).to(bundle.device)

    with torch.no_grad(), hook_residual_stream(bundle.model, range(bundle.arch.n_layers)) as hook:
        outputs = bundle.model(**tokens)

    return hook.get_all(), outputs.logits


def compute_perplexity(bundle: ModelBundle, texts: list[str]) -> float:
    """Compute perplexity on a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        input_ids = tokenize(bundle.tokenizer, text, bundle.device)
        with torch.no_grad():
            outputs = bundle.model(input_ids, labels=input_ids)
        total_loss += outputs.loss.item() * input_ids.shape[1]
        total_tokens += input_ids.shape[1]

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def select_reserved_dimension(bundle: ModelBundle, n_samples: int = 50) -> int:
    """
    Select the residual stream dimension with lowest variance across random inputs.
    This dimension is least used by the model, so perturbing it is least disruptive.
    """
    print("Selecting reserved dimension (lowest variance across inputs)...")

    all_acts = []
    for _ in range(n_samples):
        random_ids = torch.randint(0, bundle.arch.vocab_size, (1, 32), device=bundle.device)
        with torch.no_grad(), hook_residual_stream(bundle.model, [bundle.arch.n_layers - 1]) as hook:
            bundle.model(random_ids)
        acts = hook.get(bundle.arch.n_layers - 1).mean(dim=1)  # (1, d_model)
        all_acts.append(acts)

    all_acts = torch.cat(all_acts, dim=0)  # (n_samples, d_model)
    variances = all_acts.var(dim=0)  # (d_model,)

    dim_r = variances.argmin().item()
    print(f"Reserved dimension: {dim_r} (variance: {variances[dim_r]:.6f})")
    return dim_r


def free_memory():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
