"""Lightweight, deliberately lenient text-quality filtering."""

import re
from dataclasses import dataclass, field


@dataclass
class QualityResult:
    """The outcome of checking one document."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


class Heuristics:
    """Apply a small, lenient subset of Gopher/C4-style quality checks.

    Thresholds are constructor arguments so a caller can tighten or relax the
    filter without changing the implementation.  A document is rejected only
    when it violates one or more checks; use :meth:`evaluate` to inspect why.
    """

    _WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
    _ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
    _LOREM_RE = re.compile(r"\blorem ipsum\b", re.IGNORECASE)
    _CODE_RE = re.compile(
        r"(?:\b(?:function|const|var)\s+\w+|</?(?:script|style)\b|"
        r"\b(?:document\.write|javascript:))",
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 100_000,
        min_alpha_ratio: float = 0.45,
        max_symbol_ratio: float = 0.30,
        max_repeated_line_ratio: float = 0.50,
        max_code_ratio: float = 0.30,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.min_alpha_ratio = min_alpha_ratio
        self.max_symbol_ratio = max_symbol_ratio
        self.max_repeated_line_ratio = max_repeated_line_ratio
        self.max_code_ratio = max_code_ratio
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        if self.min_words < 0 or self.max_words < self.min_words:
            raise ValueError("word limits must satisfy 0 <= min_words <= max_words")
        for name in (
            "min_alpha_ratio",
            "max_symbol_ratio",
            "max_repeated_line_ratio",
            "max_code_ratio",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def evaluate(self, text: str) -> QualityResult:
        """Return a decision, rejection reasons, and useful quality metrics."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        stripped = text.strip()
        if not stripped:
            return QualityResult(False, ["empty document"], {"word_count": 0.0})

        words = self._WORD_RE.findall(stripped)
        word_count = len(words)
        non_space = [char for char in stripped if not char.isspace()]
        alpha_ratio = (
            sum(bool(self._ALPHA_RE.match(char)) for char in non_space)
            / len(non_space)
        )
        symbol_ratio = (
            sum(not char.isalnum() and char not in ".,!?;:'\"-()[]" for char in non_space)
            / len(non_space)
        )

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        repeated_line_ratio = self._repeated_line_ratio(lines)
        code_lines = sum(bool(self._CODE_RE.search(line)) for line in lines)
        code_ratio = code_lines / len(lines) if lines else 0.0

        metrics = {
            "word_count": float(word_count),
            "alpha_ratio": alpha_ratio,
            "symbol_ratio": symbol_ratio,
            "repeated_line_ratio": repeated_line_ratio,
            "code_ratio": code_ratio,
        }
        reasons: list[str] = []

        if word_count < self.min_words:
            reasons.append(f"too few words ({word_count} < {self.min_words})")
        if word_count > self.max_words:
            reasons.append(f"too many words ({word_count} > {self.max_words})")
        if alpha_ratio < self.min_alpha_ratio:
            reasons.append(f"low alphabetic-character ratio ({alpha_ratio:.2f})")
        if symbol_ratio > self.max_symbol_ratio:
            reasons.append(f"high symbol ratio ({symbol_ratio:.2f})")
        if repeated_line_ratio > self.max_repeated_line_ratio:
            reasons.append(f"too many repeated lines ({repeated_line_ratio:.2f})")
        if code_ratio > self.max_code_ratio:
            reasons.append(f"looks predominantly like code/markup ({code_ratio:.2f})")
        if self._LOREM_RE.search(stripped):
            reasons.append("contains placeholder text")

        return QualityResult(not reasons, reasons, metrics)

    def is_quality(self, text: str) -> bool:
        """Return whether ``text`` passes all enabled quality checks."""
        return self.evaluate(text).passed

    def filter(self, documents: list[str]) -> list[str]:
        """Return documents that pass the quality checks, preserving order."""
        return [document for document in documents if self.is_quality(document)]

    @staticmethod
    def _repeated_line_ratio(lines: list[str]) -> float:
        if len(lines) < 2:
            return 0.0
        normalized = [re.sub(r"\s+", " ", line).casefold() for line in lines]
        repeated = len(normalized) - len(set(normalized))
        return repeated / len(normalized)