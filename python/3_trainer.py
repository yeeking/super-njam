#!/usr/bin/env python3
"""Stage 3 CLI for a single training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.training_tools import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a llama-compatible NJam model.")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Input JSONL corpus of NJamV3 solos for training. Expected: an existing file exported by the language stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, TensorBoard logs, tokenizer files, and sample outputs. Expected: a writable directory path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size per optimization step. Expected range: 1-64 depending on memory; 1-8 is typical on small local runs.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Maximum token sequence length used for training chunks. Expected range: 64-4096; larger values require more memory.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer decoder layers. Expected range: 2-32 for this project; higher values increase capacity and training cost.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Transformer hidden width. Expected range: 64-2048; it should be divisible by --num-heads.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads. Expected range: 1-32; must divide --hidden-size evenly.",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=256,
        help="Feed-forward network width inside each transformer block. Expected range: at least --hidden-size and commonly 2x-8x hidden size.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Maximum number of full passes over the training set. Expected range: 1-200 depending on corpus size and overfitting risk.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate. Expected range: roughly 1e-5 to 1e-3; 1e-4 to 5e-4 is a reasonable starting band.",
    )
    parser.add_argument(
        "--sample-prompt-ratio",
        type=float,
        default=0.35,
        help="Fraction of each held-out sample kept as prompt before generation. Expected range: 0.05-0.95, where 0.35 means keep 35 percent as context.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=2,
        help="Number of held-out prompt continuations to render during evaluation. Expected range: 1-32 for routine runs.",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=Path("soundfonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"),
        help="Soundfont file used to render MIDI previews to audio. Expected: a readable .sf2 path. Defaults to the higher-quality local training soundfont.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="saxophone",
        help="Instrument patch used for rendered training samples. Expected values: names such as saxophone, alto_sax, tenor_sax, piano, trumpet, or clarinet. Default: saxophone.",
    )
    args = parser.parse_args()
    config = TrainConfig(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        sample_prompt_ratio=args.sample_prompt_ratio,
        sample_limit=args.sample_limit,
        soundfont_path=args.soundfont,
        render_instrument=args.instrument,
    )
    summary = run_training(config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
