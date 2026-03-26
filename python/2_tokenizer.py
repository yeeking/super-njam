#!/usr/bin/env python3
"""Stage 2 CLI for tokenizer experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.tokenizer_tools import compare_tokenizers


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NJamV3 tokenizer strategies.")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Input JSONL corpus of NJamV3 solos to analyze. Expected: an existing corpus file produced by the language export stage.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory where tokenizer models, reports, and comparison artifacts are written. Expected: a writable directory path.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Target vocabulary size for trainable tokenizers. Expected range: roughly 128-8192; 256-2048 is a sensible range for current experiments.",
    )
    args = parser.parse_args()
    reports = compare_tokenizers(args.corpus, args.out, vocab_size=args.vocab_size)
    print(json.dumps([report.__dict__ for report in reports], indent=2))


if __name__ == "__main__":
    main()
