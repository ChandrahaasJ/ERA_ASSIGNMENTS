"""BPE tokenizer with language-specific splitting for English, Telugu, and Hindi."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Sample corpus: 5 lines each for Telugu, English, and Hindi
# ---------------------------------------------------------------------------

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

# Unicode ranges for Telugu (U+0C00–U+0C7F) and Devanagari (U+0900–U+097F)
_TELUGU_CONSONANTS = set(chr(c) for c in range(0x0C15, 0x0C3A))
_TELUGU_VOWELS = set(chr(c) for c in range(0x0C05, 0x0C15))
_TELUGU_VIRAMA = "\u0C4D"
_TELUGU_MATRAS = set(chr(c) for c in list(range(0x0C3E, 0x0C4D)) + [0x0C55, 0x0C56])
_TELUGU_ANUSVARA = "\u0C02"

_DEVANAGARI_CONSONANTS = set(chr(c) for c in list(range(0x0915, 0x093A)) + list(range(0x0958, 0x0960)))
_DEVANAGARI_VOWELS = set(chr(c) for c in range(0x0905, 0x0914))
_DEVANAGARI_VIRAMA = "\u094D"
_DEVANAGARI_MATRAS = set(chr(c) for c in range(0x093E, 0x094F))
_DEVANAGARI_CHANDRABINDHU = "\u0901"
_DEVANAGARI_ANUSVARA = "\u0902"
_DEVANAGARI_VISARGA = "\u0903"
_DEVANAGARI_NUKTA = "\u093C"


def _is_telugu_consonant(ch: str) -> bool:
    return ch in _TELUGU_CONSONANTS


def _is_telugu_vowel(ch: str) -> bool:
    return ch in _TELUGU_VOWELS


def _is_devanagari_consonant(ch: str) -> bool:
    return ch in _DEVANAGARI_CONSONANTS


def _is_devanagari_vowel(ch: str) -> bool:
    return ch in _DEVANAGARI_VOWELS


def _split_mathrawise_word(word: str, script: str) -> list[str]:
    """Split a single word into grapheme clusters (consonant + matras/viramas)."""
    if not word:
        return []

    if script == "te":
        is_consonant = _is_telugu_consonant
        is_vowel = _is_telugu_vowel
        virama = _TELUGU_VIRAMA
        matras = _TELUGU_MATRAS
        chandrabindhu = {_TELUGU_ANUSVARA}
    else:
        is_consonant = _is_devanagari_consonant
        is_vowel = _is_devanagari_vowel
        virama = _DEVANAGARI_VIRAMA
        matras = _DEVANAGARI_MATRAS
        chandrabindhu = {_DEVANAGARI_CHANDRABINDHU, _DEVANAGARI_ANUSVARA, _DEVANAGARI_VISARGA}

    clusters: list[str] = []
    i = 0
    while i < len(word):
        ch = word[i]
        cluster = ch
        i += 1

        if is_consonant(ch) or is_vowel(ch):
            # Absorb virama + consonant chains (conjuncts): e.g. క్ష, ప్ర
            while i < len(word) and word[i] == virama:
                if i + 1 < len(word) and is_consonant(word[i + 1]):
                    cluster += word[i] + word[i + 1]
                    i += 2
                else:
                    cluster += word[i]
                    i += 1

            # Absorb vowel signs (matras) and chandrabindhu/anusvara
            while i < len(word) and (word[i] in matras or word[i] in chandrabindhu):
                cluster += word[i]
                i += 1

            # Hindi nukta attaches to preceding consonant
            if script == "hi" and i < len(word) and word[i] == _DEVANAGARI_NUKTA:
                cluster += word[i]
                i += 1
        else:
            # Punctuation or other symbols stay as individual clusters
            pass

        clusters.append(cluster)

    return clusters


def _split_charwise(text: str) -> list[str]:
    """Split text character by character, preserving word boundaries as spaces."""
    tokens: list[str] = []
    for word in text.split():
        tokens.extend(list(word))
        tokens.append(" ")
    if tokens and tokens[-1] == " ":
        tokens.pop()
    return tokens


def split_charwise_telugu(text: str) -> list[str]:
    """Split Telugu text character-wise (ignoring matra/virama clustering)."""
    return _split_charwise(text)


def split_mathrawise_telugu(text: str) -> list[str]:
    """Split Telugu text into grapheme clusters (consonant + matras/viramas)."""
    tokens: list[str] = []
    for word in text.split():
        tokens.extend(_split_mathrawise_word(word, "te"))
        tokens.append(" ")
    if tokens and tokens[-1] == " ":
        tokens.pop()
    return tokens


def split_charwise_hindi(text: str) -> list[str]:
    """Split Hindi text character-wise (ignoring matra/virama clustering)."""
    return _split_charwise(text)


def split_mathrawise_hindi(text: str) -> list[str]:
    """Split Hindi text into grapheme clusters (consonant + matras/viramas/chandrabindhu)."""
    tokens: list[str] = []
    for word in text.split():
        tokens.extend(_split_mathrawise_word(word, "hi"))
        tokens.append(" ")
    if tokens and tokens[-1] == " ":
        tokens.pop()
    return tokens


def split_charwise_english(text: str) -> list[str]:
    """Split English text character-wise."""
    return _split_charwise(text)


# ---------------------------------------------------------------------------
# BPE algorithm
# ---------------------------------------------------------------------------

SplitFn = Callable[[str], list[str]]


@dataclass
class BPEResult:
    vocabulary: dict[str, int]
    merges: list[tuple[str, str]]
    tokenized_corpus: list[list[str]] = field(default_factory=list)


def _get_pair_counts(corpus: list[list[str]]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
    return pairs


def _merge_pair(corpus: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
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


def _build_vocabulary(corpus: list[list[str]]) -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    for word in corpus:
        for token in word:
            vocabulary[token] = vocabulary.get(token, 0) + 1
    return vocabulary


def bpe(
    lines: list[str],
    split_fn: SplitFn,
    vocab_size: int = 65,
) -> BPEResult:
    """
    Train a BPE tokenizer on the given lines using the provided split function.

    Starting from the split-function tokens, repeatedly merges the most frequent
    adjacent pair. When the initial vocabulary is smaller than ``vocab_size``,
    merging continues until the vocabulary reaches that size. When the initial
    vocabulary is already larger, up to ``vocab_size`` merge operations are still
    learned before the final vocabulary is trimmed to the top ``vocab_size`` tokens
    by frequency.

    Args:
        lines: Corpus lines to train on.
        split_fn: Language-specific splitting function (charwise or mathrawise).
        vocab_size: Target vocabulary size (default 65).

    Returns:
        BPEResult with vocabulary, merge rules, and tokenized corpus.
    """
    corpus: list[list[str]] = []
    for line in lines:
        tokens = split_fn(line)
        if tokens:
            corpus.append(tokens)

    vocabulary = _build_vocabulary(corpus)
    merges: list[tuple[str, str]] = []
    initial_vocab_size = len(vocabulary)

    while len(vocabulary) < vocab_size:
        pair_counts = _get_pair_counts(corpus)
        if not pair_counts:
            break

        best_pair = pair_counts.most_common(1)[0][0]
        corpus = _merge_pair(corpus, best_pair)
        merges.append(best_pair)
        vocabulary = _build_vocabulary(corpus)

    if initial_vocab_size >= vocab_size:
        while len(merges) < vocab_size:
            pair_counts = _get_pair_counts(corpus)
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            corpus = _merge_pair(corpus, best_pair)
            merges.append(best_pair)

        vocabulary = _build_vocabulary(corpus)

    if len(vocabulary) > vocab_size:
        vocabulary = dict(
            sorted(vocabulary.items(), key=lambda item: (-item[1], item[0]))[:vocab_size]
        )

    return BPEResult(vocabulary=vocabulary, merges=merges, tokenized_corpus=corpus)


class Tokenizer:
    """Tokenizer that supports BPE with language-specific splitting strategies."""

    SPLIT_FUNCTIONS: dict[str, dict[str, SplitFn]] = {
        "te": {
            "charwise": split_charwise_telugu,
            "mathrawise": split_mathrawise_telugu,
        },
        "hi": {
            "charwise": split_charwise_hindi,
            "mathrawise": split_mathrawise_hindi,
        },
        "en": {
            "charwise": split_charwise_english,
        },
    }

    def __init__(self, vocab_size: int = 65):
        self.vocab_size = vocab_size
        self.results: dict[str, BPEResult] = {}

    def split_charwise(self, text: str, language: str) -> list[str]:
        """Split text character-wise for the given language (te, hi, or en)."""
        if language not in self.SPLIT_FUNCTIONS:
            raise ValueError(f"Unsupported language: {language}")
        return self.SPLIT_FUNCTIONS[language]["charwise"](text)

    def split_mathrawise(self, text: str, language: str) -> list[str]:
        """Split Telugu or Hindi text into grapheme clusters (matras/viramas/chandrabindhu)."""
        if language not in ("te", "hi"):
            raise ValueError("mathrawise splitting is only supported for Telugu (te) and Hindi (hi)")
        return self.SPLIT_FUNCTIONS[language]["mathrawise"](text)

    def train(
        self,
        telugu_lines: list[str] | None = None,
        english_lines: list[str] | None = None,
        hindi_lines: list[str] | None = None,
        split_mode: str = "mathrawise",
    ) -> dict[str, BPEResult]:
        """
        Train BPE on all provided lines using appropriate split functions.

        English is always split charwise. Telugu and Hindi use the chosen
        split_mode ('charwise' or 'mathrawise').
        """
        telugu_lines = telugu_lines or TELUGU_LINES
        english_lines = english_lines or ENGLISH_LINES
        hindi_lines = hindi_lines or HINDI_LINES

        te_fn = self.SPLIT_FUNCTIONS["te"][split_mode]
        hi_fn = self.SPLIT_FUNCTIONS["hi"][split_mode]
        en_fn = self.SPLIT_FUNCTIONS["en"]["charwise"]

        self.results = {
            "telugu": bpe(telugu_lines, te_fn, self.vocab_size),
            "english": bpe(english_lines, en_fn, self.vocab_size),
            "hindi": bpe(hindi_lines, hi_fn, self.vocab_size),
            "combined": bpe(
                telugu_lines + english_lines + hindi_lines,
                _combined_split_fn(split_mode),
                self.vocab_size,
            ),
        }
        return self.results

    def get_vocabulary(self, language: str = "combined") -> dict[str, int]:
        if language not in self.results:
            raise KeyError(f"No results for language '{language}'. Call train() first.")
        return self.results[language].vocabulary


def _combined_split_fn(split_mode: str) -> SplitFn:
    """Return a split function that routes lines to the correct language splitter."""

    def split_line(line: str) -> list[str]:
        if _contains_telugu(line):
            return split_mathrawise_telugu(line) if split_mode == "mathrawise" else split_charwise_telugu(line)
        if _contains_devanagari(line):
            return split_mathrawise_hindi(line) if split_mode == "mathrawise" else split_charwise_hindi(line)
        return split_charwise_english(line)

    return split_line


def _contains_telugu(text: str) -> bool:
    return any("\u0C00" <= ch <= "\u0C7F" for ch in text)


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


if __name__ == "__main__":
    print("=" * 60)
    print("Splitting examples (Telugu)")
    print("=" * 60)
    sample_te = "తెలుగు"
    print(f"Word: {sample_te}")
    print(f"  charwise:    {split_charwise_telugu(sample_te)}")
    print(f"  mathrawise:  {split_mathrawise_telugu(sample_te)}")

    sample_conjunct = "క్షమించండి"
    print(f"\nWord: {sample_conjunct}")
    print(f"  charwise:    {split_charwise_telugu(sample_conjunct)}")
    print(f"  mathrawise:  {split_mathrawise_telugu(sample_conjunct)}")

    print("\n" + "=" * 60)
    print("Splitting examples (Hindi)")
    print("=" * 60)
    sample_hi = "प्रेम"
    print(f"Word: {sample_hi}")
    print(f"  charwise:    {split_charwise_hindi(sample_hi)}")
    print(f"  mathrawise:  {split_mathrawise_hindi(sample_hi)}")

    print("\n" + "=" * 60)
    print("BPE Training (vocab_size=65, all 15 lines)")
    print("=" * 60)

    for mode in ("charwise", "mathrawise"):
        print(f"\n--- Split mode: {mode} ---")
        split_fn = _combined_split_fn(mode)
        result = bpe(ALL_LINES, split_fn, vocab_size=65)
        print(f"Vocabulary size: {len(result.vocabulary)}")
        print(f"Number of merges: {len(result.merges)}")
        top_tokens = sorted(result.vocabulary.items(), key=lambda x: -x[1])[:10]
        print(f"Top 10 tokens: {top_tokens}")
        if result.merges:
            print(f"First 5 merges: {result.merges[:5]}")
