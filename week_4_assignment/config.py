from dataclasses import dataclass

@dataclass
class Config:
    SHINGLE_LENGTH: int = 5
    OVERLAP: int = SHINGLE_LENGTH - 1
