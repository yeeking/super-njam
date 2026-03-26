#!/usr/bin/env python3
"""Stage 4 CLI for HF -> GGUF export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.export_tools import ExportConfig, export_hf_to_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained Hugging Face checkpoint to GGUF.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the trained Hugging Face model and tokenizer files to export. Expected: a readable model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the GGUF export is written. Expected: a writable directory path.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path("libs/llama.cpp"),
        help="Path to the local llama.cpp checkout used for conversion scripts. Expected: a directory containing the llama.cpp converter utilities.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="model-f16.gguf",
        help="Filename for the exported GGUF model. Expected: a .gguf filename such as model-f16.gguf or model-f32.gguf.",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="f16",
        help="Numeric precision for the exported GGUF weights. Expected values depend on llama.cpp conversion support; common choices are f16 and f32.",
    )
    args = parser.parse_args()
    path = export_hf_to_gguf(
        ExportConfig(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            llama_cpp_dir=args.llama_cpp_dir,
            outfile=args.outfile,
            outtype=args.outtype,
        )
    )
    print(json.dumps({"gguf_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
