from utils import Utils
from config import Config


class Dedup:

    def __init__(self):
        self.utils = Utils()
        self._minhash_coefficients: list[tuple[int, int]] | None = None

    def dedup(self, folder_path):
        """
        Perform MinHash + LSH to dedupe all the docs in the folder
        """
        shingles = self.utils.get_shingles(Config.SHINGLE_LENGTH, folder_path, Config.OVERLAP)
        return shingles

    def minhash(self, doc_a: str, doc_b: str, num_hashes: int = 100) -> float:
        """
            - get shingles for each doc
            - hash each shingle, then find the most minimum hash value for each doc to represent it
            - compute multiple min hashes for each doc using number theory (ax+b mod p)
            - to find the probability of similarity, we can use the Jaccard similarity coefficient
        """
        shingles_a = set(
            self.utils.get_shingles(Config.SHINGLE_LENGTH, doc_a, Config.OVERLAP)
        )
        shingles_b = set(
            self.utils.get_shingles(Config.SHINGLE_LENGTH, doc_b, Config.OVERLAP)
        )

        if not shingles_a and not shingles_b:
            return 1.0
        if not shingles_a or not shingles_b:
            return 0.0

        sig_a = self._minhash_signature(shingles_a, num_hashes)
        sig_b = self._minhash_signature(shingles_b, num_hashes)
        matches = sum(a == b for a, b in zip(sig_a, sig_b))
        return matches / num_hashes

    def _minhash_coefficients_for(self, num_hashes: int) -> list[tuple[int, int]]:
        if (
            self._minhash_coefficients is not None
            and len(self._minhash_coefficients) >= num_hashes
        ):
            return self._minhash_coefficients[:num_hashes]

        prime = Config.PRIME
        coefficients: list[tuple[int, int]] = []
        for i in range(num_hashes):
            a = (1103515245 * (i + 1) + 12345) % prime or 1
            b = (1664525 * (i + 1) + 1013904223) % prime
            coefficients.append((a, b))
        self._minhash_coefficients = coefficients
        return coefficients

    def _minhash_signature(self, shingles: set[str], num_hashes: int) -> list[int]:
        prime = self._PRIME
        coefficients = self._minhash_coefficients_for(num_hashes)
        signature: list[int] = []

        for a, b in coefficients:
            min_hash = prime
            for shingle in shingles:
                x = self.utils.hash(shingle) % prime
                candidate = (a * x + b) % prime
                if candidate < min_hash:
                    min_hash = candidate
            signature.append(min_hash)

        return signature

    def lsh(self, minhashes:list[int]):
        pass