"""Micro-transformer: a small decoder-only (GPT-style, pre-LayerNorm) model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int  # maximum input length in tokens
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 32
    mlp_ratio: int = 4
    dropout: float = 0.0
    tie_weights: bool = True  # share the output head with the token embedding


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.attn_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp_in = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model)
        self.mlp_out = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = (
            t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            for t in self.qkv(self.ln1(x)).chunk(3, dim=-1)
        )
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.attn_out(attn.transpose(1, 2).reshape(B, T, C)))
        x = x + self.drop(self.mlp_out(F.gelu(self.mlp_in(self.ln2(x)))))
        return x


class MicroTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError(f"d_model={cfg.d_model} not divisible by n_head={cfg.n_head}")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Token ids [B, T] -> next-token logits [B, T, vocab_size]."""
        T = idx.shape[1]
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def count_params(model: MicroTransformer) -> tuple[int, int]:
    """(num_params, num_params_non_embedding), per docs/metrics.md section 4.

    num_params is every trainable parameter, with a tied head counted once.
    The non-embedding count leaves out the token embedding, the position
    embedding and the output head, so it is the same whether or not the head
    is tied.
    """
    # parameters() yields a shared tensor once, so tying is handled already.
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding = {model.tok_emb.weight, model.pos_emb.weight, model.head.weight}
    non_embedding = total - sum(p.numel() for p in embedding if p.requires_grad)
    return total, non_embedding
