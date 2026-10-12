"""Tokenizers. Only the per-character one exists so far."""


class CharTokenizer:
    """One token per character, ids assigned in the order of `chars`."""

    def __init__(self, chars: str):
        if len(set(chars)) != len(chars):
            raise ValueError(f"chars has repeated characters: {chars!r}")
        self.chars = chars
        self._ids = {c: i for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self._ids[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)
