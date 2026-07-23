from dataclasses import dataclass

@dataclass
class Config:
    SHINGLE_LENGTH: int = 5
    OVERLAP: int = SHINGLE_LENGTH - 1
    PRIME = (1 << 61) - 1
    NUM_HASHES = 100