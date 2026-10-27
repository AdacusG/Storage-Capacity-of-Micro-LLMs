# Storage Capacity of Micro-LLMs

How many bits of information can a very small language model store, and how reliably does it store them?

This project measures the memorization capacity of micro-LLMs (roughly 10^4 to 10^7 parameters) using synthetic key-value facts, where the information content is known exactly. It also treats capacity as a probabilistic quantity by running many seeds per configuration, since models near the capacity edge show large run-to-run variation.

> **Status:** planning / early experiments. See the [Roadmap](#roadmap).

---

## Motivation

Prior work on knowledge capacity scaling suggests transformers can store on the order of 2 bits of information per parameter under favorable training conditions, and more recent work that measures memorization directly reports estimates in a similar range. (Verify exact figures and conditions against the original papers before citing; see [References](#references).) These results come mostly from larger models. The micro-LLM regime, where capacity is scarce, is much less explored.

Earlier experiments in this line of work showed that models at the edge of learning behave inconsistently across seeds. With identical hyperparameters, one run converged in 7 epochs, another in 8, and a third never converged (capped at 75 epochs). Reducing model size by one step made the data unlearnable. This suggests the capacity threshold is a probability curve, not a sharp line, so single-run results near the edge are unreliable.

## Research Questions

1. How many bits per parameter can a micro-LLM store, and how does this change with model size?
2. How sharp is the capacity threshold, and how much does it vary across seeds?
3. How do training length and the number of exposures per fact affect what is stored?
4. Does tokenization (per-character vs. small BPE vocabulary) change effective capacity?

## Method

### Synthetic data

Each dataset is a set of `N` random facts of the form `key -> value`, where keys are unique random strings and values are drawn uniformly at random from a value set of size `V`. Because values are random, they cannot be inferred or generalized, so any correct recall is memorization.

Stored information is measured as:

```
bits_stored = N * log2(V) * recall_accuracy_correction
```

with the correction handled by comparing the model's per-fact loss to the chance-level loss (a compression-style estimate: bits saved relative to an uninformed baseline). The exact estimator will be fixed and documented in `docs/metrics.md` before large sweeps.

### Models

Small decoder-only transformers, sweeping width and depth to cover a range of parameter counts. Everything else (optimizer, learning rate, schedule, batch size) is held fixed within a sweep. Optional baselines: a small RNN and a small state-space model.

### Seeds

Each configuration is run with 10+ seeds. To separate sources of randomness, seeds are split into:

- **init seed**: weight initialization
- **data seed**: shuffling order and any sampling

Varying one while fixing the other shows which source drives the variance.

### Metrics

| Metric | Meaning |
|---|---|
| Recall accuracy | Fraction of facts recalled exactly |
| Bits stored | Information stored, estimated as above |
| Bits per parameter | Bits stored divided by parameter count |
| Success rate | Fraction of seeds reaching a recall target |
| Epochs to target | Training time to hit the target (capped; failures recorded as censored) |

### Experiments

1. **Capacity vs. size:** fix `N`, sweep model size; find where recall drops.
2. **Capacity vs. dataset size:** fix model size, sweep `N`; find the number of facts at which recall degrades.
3. **Edge-of-capacity seed study:** at sizes around the threshold, run many seeds and plot success rate and epochs to target.
4. **Exposure study:** vary the number of times each fact is seen during training.
5. **Tokenization:** repeat key experiments with per-character and small-BPE tokenization, comparing in bits per character.

## Planned Figures

- Bits stored vs. parameter count, with a reference line at 2 bits/param
- Recall accuracy vs. number of facts, one curve per model size
- Success rate vs. model size at the capacity edge, with seed spread
- Epochs to target vs. dataset size (failures marked as censored)
- Per-character vs. BPE capacity comparison

## Repository Structure

```
.
├── README.md
├── docs/
│   └── metrics.md          # exact definition of bits stored, targets, caps
├── src/
│   ├── data.py             # synthetic key-value fact generation
│   ├── model.py            # micro-transformer definitions
│   ├── train.py            # training loop with init/data seed control
│   ├── eval.py             # recall and bits-stored measurement
│   └── tokenizers.py       # char-level and small-BPE tokenizers
├── configs/                # one config per sweep
├── scripts/
│   └── run_sweep.py        # launches multi-seed sweeps
├── results/                # raw per-run logs (csv/json)
└── notebooks/
    └── plots.ipynb         # figures from results/
```

## Reproducibility

- Set `torch.manual_seed` for init and a separate generator for data order.
- Use `torch.use_deterministic_algorithms(True)` where possible; note any remaining nondeterminism.
- Log every run's config, seeds, and full loss/recall curves, not just the final number.
- Report mean, std, and min/max across seeds, plus success rate.

## Roadmap

- [X] Define the bits-stored estimator and document it in `docs/metrics.md`
- [X] Implement synthetic data generator
- [X] Implement micro-transformer and training loop with separate init/data seeds
- [X] Pilot sweep: capacity vs. model size (3 seeds) to locate the edge
- [ ] Full sweep: 10+ seeds at sizes around the edge
- [ ] Exposure and dataset-size studies
- [ ] Tokenization comparison
- [ ] Write-up and figures

## References

Fill in and verify before citing:

- Allen-Zhu & Li, *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws* (2024)
- Morris et al., *How Much Do Language Models Memorize?* (2025)
- Frankle & Carbin, *The Lottery Ticket Hypothesis* (2019)

## License

TBD (MIT is a common default for research code).
