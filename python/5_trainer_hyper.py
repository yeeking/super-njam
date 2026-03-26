#!/usr/bin/env python3
"""Stage 3 CLI for a small structured hyperparameter sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.training_tools import TrainConfig, run_structured_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small structured NJam training sweep.")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Input JSONL corpus of NJamV3 solos used across all sweep runs. Expected: an existing corpus file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-run checkpoints and logs are written. Expected: a writable directory path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        required=True,
        help="Output JSON file that summarizes all sweep results. Expected: a writable file path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Base batch size applied to each sweep run. Expected range: 1-64 depending on memory; 1-8 is typical locally.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length used for every run in the sweep. Expected range: 64-4096; higher values increase memory use.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Maximum epochs per sweep run. Expected range: 1-100; keep this small for exploratory sweeps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Base learning rate used in each run. Expected range: roughly 1e-5 to 1e-3.",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        help="Optional soundfont used to render evaluation audio previews during the sweep. Expected: a readable .sf2 file path.",
    )
    args = parser.parse_args()
    config = TrainConfig(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        soundfont_path=args.soundfont,
    )
    results = run_structured_sweep(config, args.summary_out)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
