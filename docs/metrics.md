\# Metrics



Exact definitions used by every experiment in this repo. If a definition changes, bump the version at the bottom and re-run affected sweeps.



\## 1. Dataset



A dataset is `N` facts `(key\_i, value\_i)`.



\- Keys are unique random strings. They are \*\*inputs only\*\* and carry no counted information.

\- Each value is drawn independently and uniformly from a set of `V` possible values.

\- Values are random, so they cannot be inferred or generalized. Any information the model holds about them was memorized.



Chance cost of one value: `log2(V)` bits.

Maximum storable information in the dataset: `N \* log2(V)` bits.



\## 2. Bits stored (primary estimator)



The model is treated as a code for the values. Its code length on the training facts is its negative log-likelihood. Information stored is how much shorter that code is than the chance code.



```

NLL\_bits  = sum over facts of  -log2 p\_model(value\_i | key\_i)

bits\_stored = max(0,  N \* log2(V) - NLL\_bits)

```



\### Measurement rules



1\. \*\*Teacher forcing\*\*, with the key as context and the loss computed on \*\*value tokens only\*\*. Key tokens never contribute.

2\. \*\*Eval mode\*\*: dropout off, no label smoothing, no data augmentation.

3\. \*\*Units\*\*: convert nats to bits by dividing by `ln 2`.

4\. \*\*Multi-token values\*\* (e.g. per-character): sum the NLL over all value tokens to get the per-fact NLL. `V` is the number of distinct whole values, not the token vocabulary size.

5\. \*\*Clipping\*\* happens once, at the total level, not per fact. Per-fact clipping would bias the estimate upward.

6\. Computed over the \*\*training facts\*\*. Capacity here means storage of what was shown, not generalization.



\### Derived quantities



| Quantity | Formula |

|---|---|

| Bytes stored | `bits\_stored / 8` |

| Bits per parameter | `bits\_stored / num\_params` |

| Fraction of max | `bits\_stored / (N \* log2(V))` |



\## 3. Secondary metric: recall accuracy



Fraction of facts where the argmax prediction for every value token equals the true value (exact match, greedy). Report alongside bits stored. It is easier to interpret but jumps sharply, so use bits stored for curves and recall for sanity checks.



\## 4. Parameter count



`num\_params` = \*\*all trainable parameters, including embeddings and output head\*\*. Report the non-embedding count as an extra column in results files, but all headline bits/param numbers use the full count. Do not mix conventions across figures.



If the output head is tied to the input embedding, count the shared matrix once.



\## 5. Convergence and failure



Training time metrics need a target and a cap, defined per sweep in its config.



\- \*\*Target\*\*: reached when training recall accuracy >= `target\_recall` (default 0.99), evaluated every epoch.

\- \*\*Cap\*\*: maximum epochs (default 75, matching the pilot runs).

\- \*\*Success\*\*: target reached before the cap.

\- \*\*Epochs to target\*\*: epoch of first success. Runs that hit the cap are \*\*censored\*\*, recorded as `cap` with `success = False`. Never drop them or average them in as if converged.

\- \*\*Success rate\*\*: successes / seeds. Report with a binomial confidence interval (Wilson), since seed counts are small.



When the target is never reached, bits stored is still measured at the final epoch and reported as a \*\*lower bound\*\* on that run's capacity.



\## 6. Which checkpoint to measure



Bits stored is measured at the \*\*final epoch\*\* (success or cap). Also log the best value seen across epochs, but headline numbers use the final one so that results do not depend on peeking.



\## 7. Sanity checks (run before every sweep)



| Check | Expected |

|---|---|

| Untrained model on training facts | `bits\_stored \~ 0` |

| Trained model on \*\*fresh random facts\*\* (same key/value distribution, never trained on) | `bits\_stored \~ 0` |

| Model that perfectly memorizes | `bits\_stored \~ N \* log2(V)` |

| Shuffle values across keys, then train | Same ceiling, no gain from key-value structure |



If the fresh-facts check is clearly above 0, the value distribution is not uniform or keys leak information about values. Fix the generator before trusting any result.



\## 8. Comparing tokenizers



Per-token loss is not comparable across tokenizations. Use total bits per fact (as above) or bits per character of the value string. Bits stored is already tokenizer-independent because it is defined over whole values.



\## 9. Estimator caveats



\- The estimate assumes a uniform value distribution. If it is not uniform, replace `log2(V)` with the entropy of the actual value distribution.

\- Over-confident wrong predictions inflate NLL and reduce bits stored. That is intended, since a miscalibrated model is a worse code.

\- The estimate measures information in the model's outputs on the training facts. It does not prove where in the weights it is stored.

\- Capacity estimates are only meaningful in the capacity-limited regime, where more data stops increasing bits stored. Below that, the dataset is the limit.



\## Version



v0.1: initial definition.

