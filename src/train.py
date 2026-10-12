"""Train one micro-transformer on one synthetic fact dataset.

Two seeds are kept apart:
  init_seed  seeds torch's global generator: weight initialization (and
             dropout masks, if dropout is on).
  data_seed  generates the facts and, through its own torch.Generator, the
             per-epoch shuffling order.

Convergence, the cap and which checkpoint is measured follow docs/metrics.md
sections 5 and 6.

Example:
    python src/train.py --N 1000 --V 100 --d_model 32 --init_seed 0 --data_seed 0
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from data import DEFAULT_KEY_LEN, make_facts, make_fresh_facts
from eval import encode_facts, evaluate, per_fact_nll
from model import MicroTransformer, ModelConfig, count_params
from tokenizers import CharTokenizer


@dataclass(frozen=True)
class TrainConfig:
    # data
    N: int = 1000
    V: int = 100
    key_len: int = DEFAULT_KEY_LEN
    # seeds
    init_seed: int = 0
    data_seed: int = 0
    # model
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 32
    mlp_ratio: int = 4
    dropout: float = 0.0
    tie_weights: bool = True
    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    # convergence
    target_recall: float = 0.99
    max_epochs: int = 75
    device: str = "cpu"


def train(cfg: TrainConfig, verbose: bool = False) -> dict:
    # warn_only: a few CUDA ops have no deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(cfg.device)

    dataset = make_facts(cfg.N, cfg.V, cfg.data_seed, key_len=cfg.key_len)
    tokenizer = CharTokenizer(dataset.chars)
    tokens = encode_facts(dataset, tokenizer, device)
    prompt_len = len(tokenizer.encode(dataset[0].prompt))
    order_rng = torch.Generator().manual_seed(cfg.data_seed)

    torch.manual_seed(cfg.init_seed)
    model = MicroTransformer(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=tokens.shape[1] - 1,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            d_model=cfg.d_model,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            tie_weights=cfg.tie_weights,
        )
    ).to(device)
    num_params, num_params_non_embedding = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def measure(epoch: int, train_loss_bits: float | None) -> dict:
        row = {"epoch": epoch, "train_loss_bits": train_loss_bits}
        row.update(evaluate(model, tokens, prompt_len, cfg.V))
        if verbose:
            print(
                f"epoch {epoch:3d}  recall {row['recall']:.4f}  "
                f"bits {row['bits_stored']:.1f} / {dataset.max_bits:.1f}"
            )
        return row

    # Epoch 0 is the untrained model: bits_stored should be ~0 (metrics.md section 7).
    history = [measure(0, None)]
    success = False
    start_time = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(cfg.N, generator=order_rng).to(device)
        loss_nats = 0.0
        for start in range(0, cfg.N, cfg.batch_size):
            batch = tokens[order[start : start + cfg.batch_size]]
            # Same objective as the metric: value tokens only, so no capacity
            # is spent modelling the random keys.
            nll, _ = per_fact_nll(model, batch, prompt_len)
            optimizer.zero_grad(set_to_none=True)
            nll.mean().backward()
            optimizer.step()
            loss_nats += nll.detach().sum().item()

        history.append(measure(epoch, loss_nats / cfg.N / math.log(2)))
        if history[-1]["recall"] >= cfg.target_recall:
            success = True
            break

    final = history[-1]
    fresh = make_fresh_facts(dataset, cfg.data_seed + 1)
    fresh_eval = evaluate(model, encode_facts(fresh, tokenizer, device), prompt_len, cfg.V)
    return {
        "config": asdict(cfg),
        "num_params": num_params,
        "num_params_non_embedding": num_params_non_embedding,
        "max_bits": dataset.max_bits,
        "success": success,
        # Censored runs are recorded as "cap", never as a number.
        "epochs_to_target": final["epoch"] if success else "cap",
        "epochs_run": final["epoch"],
        # Headline numbers come from the final epoch. On a censored run they
        # are a lower bound on capacity.
        "recall": final["recall"],
        "nll_bits": final["nll_bits"],
        "bits_stored": final["bits_stored"],
        "bytes_stored": final["bits_stored"] / 8,
        "bits_per_param": final["bits_stored"] / num_params,
        "fraction_of_max": final["fraction_of_max"],
        "best_bits_stored": max(row["bits_stored"] for row in history),
        # Sanity check on never-trained facts: bits_stored should be ~0.
        "fresh_bits_stored": fresh_eval["bits_stored"],
        "fresh_nll_bits": fresh_eval["nll_bits"],
        "train_seconds": time.time() - start_time,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f in fields(TrainConfig):
        if f.type is bool:
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
    parser.add_argument("--out", type=Path, help="write the full result as JSON to this path")
    args = vars(parser.parse_args())
    out = args.pop("out")

    result = train(TrainConfig(**args), verbose=True)
    summary = {k: v for k, v in result.items() if k not in ("history", "config")}
    print(json.dumps(summary, indent=2))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
