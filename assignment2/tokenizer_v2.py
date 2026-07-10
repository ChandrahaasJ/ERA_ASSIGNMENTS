"""BPE tokenizer v2 with GPT-2-style regex pre-tokenization and NFC/NFD variants."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import regex as re
from tqdm import tqdm

# GPT-2 style pre-tokenizer pattern (requires the `regex` package for \p{L}/\p{N}).
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

UNK_TOKEN = "<unk>"

# Sample multilingual corpus (same lines as tokenizer.py) for demos / smoke tests.
TELUGU_LINES = [
    "తెలుగు భాష ప్రపంచంలో అత్యంత పురాతన భాషలలో ఒకటి.",
    "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని నగరం.",
    "క్షమించండి, నాకు సహాయం కావాలి.",
    "ప్రేమ మరియు స్నేహం జీవితానికి అవసరం.",
    "ఆంధ్రప్రదేశ్ దక్షిణ భారతదేశంలో ఉంది.",
]

ENGLISH_LINES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology worldwide.",
    "Natural language processing enables computers to understand text.",
    "Tokenization is the first step in building language models.",
    "Byte pair encoding creates efficient subword vocabularies.",
]

HINDI_LINES = [
    "हिन्दी भारत की राजभाषा है।",
    "दिल्ली भारत की राजधानी है।",
    "क्षमा कीजिए, मुझे मदद चाहिए।",
    "प्रेम और मित्रता जीवन के लिए आवश्यक हैं।",
    "संस्कृत भारतीय भाषाओं की जननी है।",
]

ALL_LINES = TELUGU_LINES + ENGLISH_LINES + HINDI_LINES


def _normalize(text: str, form: str) -> str:
    return unicodedata.normalize(form, text)


def _pretokenize(text: str, form: str) -> list[list[str]]:
    """Normalize, GPT-2 regex-split, then expand each chunk to characters."""
    normalized = _normalize(text, form)
    sequences: list[list[str]] = []
    for chunk in GPT2_PATTERN.findall(normalized):
        if chunk:
            sequences.append(list(chunk))
    return sequences


def _get_pair_counts(corpus: list[list[str]]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
    return pairs


def _merge_pair(corpus: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
    """Merge all non-overlapping occurrences of ``pair`` left-to-right in each sequence."""
    merged_token = pair[0] + pair[1]
    new_corpus: list[list[str]] = []
    for word in corpus:
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus


def _build_token_counts(corpus: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in corpus:
        for token in word:
            counts[token] = counts.get(token, 0) + 1
    return counts


def _train_bpe(
    corpus: list[list[str]],
    vocab_size: int,
    desc: str = "BPE merges",
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """
    Train BPE and build a token -> ID vocabulary.

    ``vocab_size`` is the final vocabulary size including ``<unk>`` at ID 0.
    Base characters are added first (frequency desc, then lexicographic).
    Each learned merge appends its merged token until the vocabulary is full.
    """
    if vocab_size < 1:
        raise ValueError("vocab_size must be at least 1")
    if not corpus:
        return {UNK_TOKEN: 0}, []

    counts = _build_token_counts(corpus)
    vocab: dict[str, int] = {UNK_TOKEN: 0}
    for token, _freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(vocab) >= vocab_size:
            break
        if token == UNK_TOKEN:
            continue
        vocab[token] = len(vocab)

    merges: list[tuple[str, str]] = []
    remaining = vocab_size - len(vocab)
    with tqdm(total=remaining, desc=desc, unit="merge", leave=True) as pbar:
        while len(vocab) < vocab_size:
            pair_counts = _get_pair_counts(corpus)
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0][0]
            corpus = _merge_pair(corpus, best_pair)
            merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in vocab:
                vocab[merged_token] = len(vocab)
                pbar.update(1)
            pbar.set_postfix(vocab=len(vocab), refresh=False)

    return vocab, merges


def _apply_bpe_merges(
    tokens: list[str],
    ranks: dict[tuple[str, str], int],
) -> list[str]:
    """
    Replay BPE merges by lowest rank until no applicable pair remains.

    On each step, find the adjacent pair with the earliest-learned rank and merge
    all non-overlapping occurrences of that pair left-to-right.
    """
    if len(tokens) < 2 or not ranks:
        return tokens

    while True:
        best_pair: tuple[str, str] | None = None
        best_rank = float("inf")
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = pair
        if best_pair is None:
            break
        tokens = _merge_pair([tokens], best_pair)[0]
        if len(tokens) < 2:
            break
    return tokens


class Tokenizer:
    """Single-class BPE tokenizer with NFC and NFD training/encoding variants."""

    def __init__(self, vocab_size: int = 65, output_dir: str | Path = ".") -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1 to reserve <unk>")
        self.vocab_size = vocab_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._vocab: dict[str, dict[str, int]] = {}
        self._merges: dict[str, list[tuple[str, str]]] = {}
        self._ranks: dict[str, dict[tuple[str, str], int]] = {}

    def _artifact_path(self, form: str) -> Path:
        return self.output_dir / f"tokenizer_{form.lower()}.json"

    def _write_artifacts(
        self,
        form: str,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
    ) -> Path:
        path = self._artifact_path(form)
        payload = {
            "merges": [list(pair) for pair in merges],
            "token_to_id": vocab,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

        key = form.lower()
        self._vocab[key] = vocab
        self._merges[key] = merges
        self._ranks[key] = {pair: i for i, pair in enumerate(merges)}
        return path

    def _load_artifacts(self, form: str) -> None:
        key = form.lower()
        if key in self._vocab and key in self._ranks:
            return

        path = self._artifact_path(form)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifacts for {form}: expected {path}. "
                f"Call generate_tokens_{key}() first."
            )

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        vocab = data["token_to_id"]
        merges = [tuple(pair) for pair in data["merges"]]

        self._vocab[key] = vocab
        self._merges[key] = merges
        self._ranks[key] = {pair: i for i, pair in enumerate(merges)}

    def _generate_tokens(self, text: str, form: str) -> Path:
        # Train on newline-separated lines; empty lines are skipped.
        lines = [line for line in text.splitlines() if line.strip()]
        corpus: list[list[str]] = []
        for line in lines:
            corpus.extend(_pretokenize(line, form))

        vocab, merges = _train_bpe(
            corpus,
            vocab_size=self.vocab_size,
            desc=f"Training {form} tokenizer",
        )
        return self._write_artifacts(form, vocab, merges)

    def generate_tokens_nfc(self, text: str) -> Path:
        """Train BPE with NFC normalization; write tokenizer_nfc.json."""
        return self._generate_tokens(text, "NFC")

    def generate_tokens_nfd(self, text: str) -> Path:
        """Train BPE with NFD normalization; write tokenizer_nfd.json."""
        return self._generate_tokens(text, "NFD")

    def _tokenize(self, text: str, form: str) -> list[int]:
        self._load_artifacts(form)
        key = form.lower()
        vocab = self._vocab[key]
        ranks = self._ranks[key]
        unk_id = vocab.get(UNK_TOKEN, 0)

        ids: list[int] = []
        for sequence in _pretokenize(text, form):
            merged = _apply_bpe_merges(sequence, ranks)
            for token in merged:
                ids.append(vocab.get(token, unk_id))
        return ids

    def tokenize_nfc(self, text: str) -> list[int]:
        """Encode ``text`` using NFC artifacts (merge replay + vocab lookup)."""
        return self._tokenize(text, "NFC")

    def tokenize_nfd(self, text: str) -> list[int]:
        """Encode ``text`` using NFD artifacts (merge replay + vocab lookup)."""
        return self._tokenize(text, "NFD")


if __name__ == "__main__":
    corpus_text = "\n".join(ALL_LINES)
    out = Path(__file__).resolve().parent / "tokenizer_v2_artifacts"
    tok = Tokenizer(vocab_size=300, output_dir=out)

    nfc_path = tok.generate_tokens_nfc(corpus_text)
    nfd_path = tok.generate_tokens_nfd(corpus_text)
    print(f"NFC tokenizer -> {nfc_path}")
    print(f"NFD tokenizer -> {nfd_path}")
    print(f"NFC vocab size: {len(tok._vocab['nfc'])}")
    print(f"NFD vocab size: {len(tok._vocab['nfd'])}")
    print(f"NFC merges: {len(tok._merges['nfc'])}")
    print(f"NFD merges: {len(tok._merges['nfd'])}")

    sample = "The quick brown fox"
    print(f"tokenize_nfc({sample!r}) -> {tok.tokenize_nfc(sample)}")
    print(f"tokenize_nfd({sample!r}) -> {tok.tokenize_nfd(sample)}")

    oov = "zzzqqq\uFFFF"
    print(f"tokenize_nfc OOV sample -> {tok.tokenize_nfc(oov)}")

    cafe = "café"
    print(f"tokenize_nfc({cafe!r}) -> {tok.tokenize_nfc(cafe)}")
    print(f"tokenize_nfd({cafe!r}) -> {tok.tokenize_nfd(cafe)}")
