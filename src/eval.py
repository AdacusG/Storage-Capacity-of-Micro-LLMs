"""Recall and bits-stored measurement, per docs/metrics.md sections 2 and 3."""

import math

import torch
import torch.nn.functional as F

from data import FactDataset
from tokenizers import CharTokenizer


def encode_facts(dataset: FactDataset, tokenizer: CharTokenizer, device="cpu") -> torch.Tensor:
    """Token ids [N, T] for every fact's text. Facts are fixed length, so no padding."""
    return torch.tensor([tokenizer.encode(f.text) for f in dataset], device=device)


def per_fact_nll(model, tokens: torch.Tensor, prompt_len: int):
    """Teacher-forced NLL of each fact's value given its key.

    `tokens` is [B, T] and its first `prompt_len` tokens are the prompt. Only
    value tokens contribute. Returns (nll in nats [B], exact-match flags [B]).
    """
    logits = model(tokens[:, :-1])[:, prompt_len - 1 :]
    targets = tokens[:, prompt_len:]
    nll = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").sum(dim=1)
    exact = (logits.argmax(dim=-1) == targets).all(dim=1)
    return nll, exact


@torch.no_grad()
def evaluate(model, tokens: torch.Tensor, prompt_len: int, V: int, batch_size: int = 4096) -> dict:
    """Bits stored and recall accuracy on the facts in `tokens`, in eval mode."""
    was_training = model.training
    model.eval()
    nll_nats = 0.0
    recalled = 0
    for start in range(0, len(tokens), batch_size):
        nll, exact = per_fact_nll(model, tokens[start : start + batch_size], prompt_len)
        nll_nats += nll.double().sum().item()
        recalled += exact.sum().item()
    model.train(was_training)

    N = len(tokens)
    max_bits = N * math.log2(V)
    nll_bits = nll_nats / math.log(2)
    # Clipped once, at the total level, never per fact.
    bits_stored = max(0.0, max_bits - nll_bits)
    return {
        "nll_bits": nll_bits,
        "bits_stored": bits_stored,
        "fraction_of_max": bits_stored / max_bits if max_bits > 0 else 0.0,
        "recall": recalled / N,
    }
