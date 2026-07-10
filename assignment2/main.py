"""Pipeline: scrape Wikipedia corpus, then train tokenizer_v2 artifacts."""

from __future__ import annotations

from pathlib import Path

from data_collector import DEFAULT_OUTPUT, collect_and_save
from tokenizer_v2 import Tokenizer

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "token_v2_artifact"
VOCAB_SIZE = 10000


def main() -> None:
    # 1. Scrape the four Wikipedia pages into scraped_india.md
    corpus_path = collect_and_save()

    # 2. Train NFC/NFD tokenizers on the scraped corpus
    text = corpus_path.read_text(encoding="utf-8")
    tok = Tokenizer(vocab_size=VOCAB_SIZE, output_dir=ARTIFACT_DIR)

    print(f"Training NFC tokenizer (vocab_size={VOCAB_SIZE})...")
    nfc_path = tok.generate_tokens_nfc(text)
    print(f"Training NFD tokenizer (vocab_size={VOCAB_SIZE})...")
    nfd_path = tok.generate_tokens_nfd(text)

    print(f"Corpus: {corpus_path} ({len(text):,} characters)")
    print(f"NFC tokenizer -> {nfc_path} (vocab={len(tok._vocab['nfc'])})")
    print(f"NFD tokenizer -> {nfd_path} (vocab={len(tok._vocab['nfd'])})")


if __name__ == "__main__":
    main()
