"""Week 4 assignment utilities."""


class utils:
    """Utility helpers for text processing."""

    def get_shingles(self, n: int, content: str, overlap: int = 0) -> list[str]:
        """
        Split content into word shingles of length up to n with optional overlap.

        Each shingle contains at most n consecutive words. Consecutive shingles
        share `overlap` words when overlap > 0. The final shingle may contain
        fewer than n words if the remaining text is shorter.
        """
        words = content.split()
        if n <= 0 or not words:
            return []

        step = n - overlap
        if step <= 0:
            raise ValueError("overlap must be smaller than n")

        shingles: list[str] = []
        i = 0
        while i < len(words):
            chunk = words[i : i + n]
            shingles.append(" ".join(chunk))
            if i + len(chunk) >= len(words):
                break
            i += step

        return shingles

    def hash(self, content: str) -> int:
        """Return an integer hash for the given content."""
        return hash(content)
