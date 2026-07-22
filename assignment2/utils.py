"""Utilities for validating a tokenizer via fertility and language stats."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import regex as re

# GPT-2 style pre-tokenizer (matches tokenizer_v2.py / visuals/index.html).
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

UNK_TOKEN = "<unk>"

LANGUAGES = ("English", "Hindi", "Telugu", "Kannada")

# Stable sample paragraphs for fertility (one per language).
FERTILITY_SAMPLES: dict[str, str] = {
    "English": (
        "India is a diverse country with a rich history of languages, "
        "cultures, and traditions that span thousands of years."
    ),
    "Hindi": (
        "भारत एक विविधतापूर्ण देश है जहाँ अनेक भाषाएँ संस्कृतियाँ "
        "और परंपराएँ हजारों वर्षों से फलती फूलती रही हैं।"
    ),
    "Telugu": (
        "భారతదేశం వైవిధ్యభరితమైన దేశం. ఇక్కడ అనేక భాషలు సంస్కృతులు "
        "మరియు సంప్రదాయాలు వేల సంవత్సరాలుగా వికసించాయి."
    ),
    "Kannada": (
        "ಭಾರತವು ವೈವಿಧ್ಯಮಯ ದೇಶ. ಇಲ್ಲಿ ಅನೇಕ ಭಾಷೆಗಳು ಸಂಸ್ಕೃತಿಗಳು "
        "ಮತ್ತು ಸಂಪ್ರದಾಯಗಳು ಸಾವಿರಾರು ವರ್ಷಗಳಿಂದ ಬೆಳೆದು ಬಂದಿವೆ."
    ),
}


def classify_script(text: str) -> str:
    """Classify a token by its first strong alphabetic script character."""
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        cp = ord(ch)
        name = unicodedata.name(ch, "")
        if "DEVANAGARI" in name or 0x0900 <= cp <= 0x097F:
            return "Hindi"
        if "TELUGU" in name or 0x0C00 <= cp <= 0x0C7F:
            return "Telugu"
        if "KANNADA" in name or 0x0C80 <= cp <= 0x0CFF:
            return "Kannada"
        if "LATIN" in name or (0x0041 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            return "English"
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            return "English"
    return "Other"


def _pair_key(left: str, right: str) -> str:
    return left + "\0" + right


def _build_ranks(merges: list[list[str] | tuple[str, str]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for index, pair in enumerate(merges):
        left, right = pair[0], pair[1]
        ranks[_pair_key(left, right)] = index
    return ranks


def _pretokenize(text: str) -> list[list[str]]:
    normalized = unicodedata.normalize("NFC", text)
    sequences: list[list[str]] = []
    for chunk in GPT2_PATTERN.findall(normalized):
        if chunk:
            sequences.append(list(chunk))
    return sequences


def _merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    left, right = pair
    merged = left + right
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
            result.append(merged)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def _apply_bpe_merges(tokens: list[str], ranks: dict[str, int]) -> list[str]:
    if len(tokens) < 2 or not ranks:
        return tokens

    while True:
        best_pair: tuple[str, str] | None = None
        best_rank = float("inf")
        for i in range(len(tokens) - 1):
            rank = ranks.get(_pair_key(tokens[i], tokens[i + 1]))
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = (tokens[i], tokens[i + 1])
        if best_pair is None:
            break
        tokens = _merge_pair(tokens, best_pair)
        if len(tokens) < 2:
            break
    return tokens


def _load_tokenizer(data_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data_or_path, dict):
        return data_or_path
    with open(data_or_path, encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str, vocab: dict[str, Any]) -> list[int]:
    """
    Encode ``text`` with NFC BPE using a tokenizer.json payload.

    ``vocab`` may be the full artifact ``{"merges": ..., "token_to_id": ...}``
    or a bare ``token_to_id`` map (no merges → character-level IDs only).
    """
    if "token_to_id" in vocab:
        token_to_id: dict[str, int] = vocab["token_to_id"]
        merges = vocab.get("merges") or []
    else:
        token_to_id = vocab  # type: ignore[assignment]
        merges = []

    ranks = _build_ranks(merges)
    unk_id = token_to_id.get(UNK_TOKEN, 0)

    ids: list[int] = []
    for sequence in _pretokenize(text):
        for token in _apply_bpe_merges(sequence, ranks):
            ids.append(token_to_id.get(token, unk_id))
    return ids


def calculate_fertility(json_path: str, text: str) -> float:
    """
    Calculate tokenizer fertility for the given text.

    Fertility is defined as the number of tokens divided by the number of words.
    For example, a string with 8 words and 12 tokens has fertility 12/8 = 1.5.
    """
    data = _load_tokenizer(json_path)
    tokens = tokenize(text, data)
    words = text.split()

    if not words:
        return 0.0

    return len(tokens) / len(words)


def language_vocab_and_merges(json_path: str | Path) -> dict[str, dict[str, int]]:
    """
    Count vocabulary entries and merges per language script.

    Returns a dict keyed by language (plus ``Other``) with ``vocab`` and ``merges``.
    """
    data = _load_tokenizer(json_path)
    token_to_id: dict[str, int] = data["token_to_id"]
    merges: list[list[str]] = data.get("merges") or []

    vocab_counts: Counter[str] = Counter()
    for token in token_to_id:
        vocab_counts[classify_script(token)] += 1

    merge_counts: Counter[str] = Counter()
    for left, right in merges:
        merge_counts[classify_script(left + right)] += 1

    result: dict[str, dict[str, int]] = {}
    for lang in (*LANGUAGES, "Other"):
        result[lang] = {
            "vocab": vocab_counts.get(lang, 0),
            "merges": merge_counts.get(lang, 0),
        }
    return result


def language_stats_report(json_path: str | Path) -> list[dict[str, Any]]:
    """
    Build table rows for English, Hindi, Telugu, and Kannada.

    Each row: language, vocab, total_vocab, merges, fertility.
    """
    path = Path(json_path)
    data = _load_tokenizer(path)
    total_vocab = len(data["token_to_id"])
    counts = language_vocab_and_merges(data)

    rows: list[dict[str, Any]] = []
    for lang in LANGUAGES:
        sample = FERTILITY_SAMPLES[lang]
        words = sample.split()
        fertility = (len(tokenize(sample, data)) / len(words)) if words else 0.0
        rows.append(
            {
                "language": lang,
                "vocab": counts[lang]["vocab"],
                "total_vocab": total_vocab,
                "merges": counts[lang]["merges"],
                "fertility": round(fertility, 3),
            }
        )
    return rows


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parent / "visuals" / "tokenizer.json"
    report = language_stats_report(default_path)
    print(f"Tokenizer: {default_path}")
    print(f"{'Language':<10} {'Vocabulary':>14} {'Merges':>8} {'Fertility':>10}")
    for row in report:
        vocab_cell = f"{row['vocab']} / {row['total_vocab']}"
        print(
            f"{row['language']:<10} {vocab_cell:>14} {row['merges']:>8} {row['fertility']:>10.3f}"
        )
