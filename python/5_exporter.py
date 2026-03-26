#!/usr/bin/env python3
"""Stage 4 CLI for exporting a Lightning checkpoint to GGUF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.export_tools import CheckpointExportConfig, export_checkpoint_to_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Lightning run checkpoint to GGUF.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint file to export. Expected: an existing Lightning .ckpt file under run_dir/checkpoints/, so the run directory can be inferred automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the GGUF file will be written. Expected: a writable directory path.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path("libs/llama.cpp"),
        help="Path to the local llama.cpp checkout containing convert_hf_to_gguf.py. Expected: an existing llama.cpp tree.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="model-f16.gguf",
        help="Output GGUF filename. Expected: a .gguf file name such as model-f16.gguf or run-q8.gguf.",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="f16",
        help="GGUF export type passed to llama.cpp. Expected values include f16 and f32.",
    )
    args = parser.parse_args()
    run_dir = args.ckpt.parent.parent
    assert run_dir.exists(), f"Could not infer run directory from checkpoint path: {args.ckpt}"
    assert (run_dir / "train_summary.json").exists(), (
        f"Inferred run directory does not look valid: {run_dir}. "
        "Expected train_summary.json next to the checkpoints directory."
    )

    output_path = export_checkpoint_to_gguf(
        CheckpointExportConfig(
            run_dir=run_dir,
            checkpoint_path=args.ckpt,
            output_dir=args.output_dir,
            llama_cpp_dir=args.llama_cpp_dir,
            outfile=args.outfile,
            outtype=args.outtype,
        )
    )
    print(json.dumps({"gguf_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
