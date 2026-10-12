"""Synthetic key-value fact generation.

Implements the dataset defined in docs/metrics.md, section 1: N facts
(key_i, value_i) where keys are unique random strings and each value is drawn
independently and uniformly from a set of V possible values.

Everything here works on plain strings so it stays tokenizer-independent.
Randomness comes from a local generator derived from `data_seed`; the global
`random` state and torch's seed (the init seed) are never touched.
"""

import math
import random
import string
from collections import Counter
from dataclasses import dataclass, replace

DEFAULT_ALPHABET = string.ascii_lowercase
DEFAULT_KEY_LEN = 8
# Separates key from value in the model input. Must not be in either alphabet.
SEP = ":"


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    value_id: int  # index of `value` in the dataset's value set, in [0, V)

    @property
    def prompt(self) -> str:
        """Context the model is given. Never contributes to the loss."""
        return self.key + SEP

    @property
    def text(self) -> str:
        """Full training string. Loss is computed on the part after `prompt`."""
        return self.prompt + self.value


@dataclass(frozen=True)
class FactDataset:
    facts: tuple[Fact, ...]
    values: tuple[str, ...]  # the V possible values; values[value_id] is the string
    data_seed: int
    key_alphabet: str = DEFAULT_ALPHABET
    value_alphabet: str = DEFAULT_ALPHABET

    def __len__(self) -> int:
        return len(self.facts)

    def __getitem__(self, i: int) -> Fact:
        return self.facts[i]

    def __iter__(self):
        return iter(self.facts)

    @property
    def N(self) -> int:
        return len(self.facts)

    @property
    def V(self) -> int:
        return len(self.values)

    @property
    def key_len(self) -> int:
        return len(self.facts[0].key)

    @property
    def value_len(self) -> int:
        return len(self.values[0])

    @property
    def chance_bits_per_fact(self) -> float:
        """Chance cost of one value: log2(V) bits."""
        return math.log2(self.V)

    @property
    def max_bits(self) -> float:
        """Maximum storable information in the dataset: N * log2(V) bits."""
        return self.N * self.chance_bits_per_fact

    @property
    def empirical_value_entropy_bits(self) -> float:
        """Entropy of the realized value distribution, in bits per fact.

        Slightly below log2(V) for finite N even when the generator is uniform.
        A large gap means the generator is broken (see metrics.md, sections 7, 9).
        """
        counts = Counter(f.value_id for f in self.facts)
        return -sum(c / self.N * math.log2(c / self.N) for c in counts.values())

    @property
    def chars(self) -> str:
        """Every character that can appear in a fact's text, sorted."""
        return "".join(sorted(set(self.key_alphabet + self.value_alphabet + SEP)))


def min_value_len(V: int, alphabet: str = DEFAULT_ALPHABET) -> int:
    """Shortest fixed length that gives at least V distinct strings."""
    length = 1
    while len(alphabet) ** length < V:
        length += 1
    return length


def make_value_set(
    V: int, alphabet: str = DEFAULT_ALPHABET, value_len: int | None = None
) -> tuple[str, ...]:
    """The V possible values: index i written in base len(alphabet), fixed width.

    Deterministic, so every dataset with the same (V, alphabet, value_len)
    shares one value set regardless of seed.
    """
    if V < 1:
        raise ValueError(f"V must be >= 1, got {V}")
    _check_alphabet(alphabet)
    base = len(alphabet)
    if value_len is None:
        value_len = min_value_len(V, alphabet)
    if base**value_len < V:
        raise ValueError(
            f"value_len={value_len} over {base} characters gives only "
            f"{base**value_len} strings, need V={V}"
        )
    values = []
    for i in range(V):
        chars = []
        for _ in range(value_len):
            i, digit = divmod(i, base)
            chars.append(alphabet[digit])
        values.append("".join(reversed(chars)))
    return tuple(values)


def make_facts(
    N: int,
    V: int,
    data_seed: int,
    *,
    key_len: int = DEFAULT_KEY_LEN,
    key_alphabet: str = DEFAULT_ALPHABET,
    value_alphabet: str = DEFAULT_ALPHABET,
    value_len: int | None = None,
    exclude_keys: frozenset[str] | set[str] = frozenset(),
) -> FactDataset:
    """Generate N facts with unique random keys and i.i.d. uniform values.

    Keys and values use independent random streams, so keys carry no
    information about values and changing V does not change the keys.
    `exclude_keys` are never generated (used to keep fresh facts disjoint
    from training facts).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    _check_alphabet(key_alphabet)
    values = make_value_set(V, value_alphabet, value_len)

    if len(key_alphabet) ** key_len < 2 * (N + len(exclude_keys)):
        raise ValueError(
            f"key_len={key_len} over {len(key_alphabet)} characters is too small "
            f"for {N} unique keys; increase key_len"
        )
    key_rng = random.Random(f"{data_seed}/keys")
    keys: list[str] = []
    seen = set(exclude_keys)
    while len(keys) < N:
        key = "".join(key_rng.choices(key_alphabet, k=key_len))
        if key not in seen:
            seen.add(key)
            keys.append(key)

    value_rng = random.Random(f"{data_seed}/values")
    facts = []
    for key in keys:
        value_id = value_rng.randrange(V)
        facts.append(Fact(key, values[value_id], value_id))

    return FactDataset(tuple(facts), values, data_seed, key_alphabet, value_alphabet)


def make_fresh_facts(train: FactDataset, data_seed: int, N: int | None = None) -> FactDataset:
    """Fresh random facts from the same key/value distribution as `train`.

    For the sanity check in metrics.md, section 7: a model trained on `train`
    should score bits_stored ~ 0 on these. Keys are disjoint from `train`'s.
    """
    if data_seed == train.data_seed:
        raise ValueError("fresh facts need a data_seed different from the training set's")
    return make_facts(
        N if N is not None else train.N,
        train.V,
        data_seed,
        key_len=train.key_len,
        key_alphabet=train.key_alphabet,
        value_alphabet=train.value_alphabet,
        value_len=train.value_len,
        exclude_keys={f.key for f in train.facts},
    )


def shuffle_values(dataset: FactDataset, seed: int) -> FactDataset:
    """Permute values across keys, keeping the keys and the value multiset.

    For the sanity check in metrics.md, section 7: training on the result
    should reach the same ceiling as training on the original.
    """
    order = list(range(dataset.N))
    random.Random(f"{seed}/shuffle").shuffle(order)
    facts = tuple(
        Fact(fact.key, dataset.facts[j].value, dataset.facts[j].value_id)
        for fact, j in zip(dataset.facts, order)
    )
    return replace(dataset, facts=facts)


def _check_alphabet(alphabet: str) -> None:
    if not alphabet:
        raise ValueError("alphabet must be non-empty")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(f"alphabet has repeated characters: {alphabet!r}")
    if SEP in alphabet:
        raise ValueError(f"alphabet must not contain the separator {SEP!r}")
