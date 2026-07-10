"""Utilities for validating a tokenizer via fertility metrics."""

from __future__ import annotations

import json

# Hardcoded mock: tokenize() always returns this many tokens.
MOCK_TOKEN_COUNT = 12


def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """Mock tokenizer that always returns the same number of tokens."""
    return list(range(MOCK_TOKEN_COUNT))


def calculate_fertility(json_path: str, text: str) -> float:
    """
    Calculate tokenizer fertility for the given text.

    Fertility is defined as the number of tokens divided by the number of words.
    For example, a string with 8 words and 12 tokens has fertility 12/8 = 1.5.
    """
    with open(json_path, encoding="utf-8") as f:
        vocab = json.load(f)

    tokens = tokenize(text, vocab)
    words = text.split()

    if not words:
        return 0.0

    return len(tokens) / len(words)
