#!/usr/bin/env python3
"""Stage 4 CLI for quantization and llama.cpp benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.export_tools import bench_gguf, quantize_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and benchmark GGUF models.")
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path("libs/llama.cpp"),
        help="Path to the local llama.cpp checkout used for quantization and benchmarking. Expected: a built llama.cpp tree with the required tools available.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Input GGUF model to quantize or benchmark. Expected: an existing .gguf file.",
    )
    parser.add_argument(
        "--quant",
        type=str,
        help="Optional quantization preset to apply before benchmarking. Expected values depend on llama.cpp, for example q8_0, q6_k, q5_k_m, or q4_k_m.",
    )
    parser.add_argument(
        "--quant-out",
        type=Path,
        help="Output path for the quantized GGUF file. Required when --quant is set. Expected: a writable .gguf path.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text prompt file used for benchmarking generation. Expected: a readable NJamV3 or plain text prompt file.",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=64,
        help="Number of tokens to generate during benchmarking. Expected range: 1-4096; 16-256 is a practical smoke-test range.",
    )
    args = parser.parse_args()

    model_path = args.model
    payload = {}
    if args.quant:
        assert args.quant_out is not None, "--quant-out is required when --quant is set."
        model_path = quantize_gguf(args.llama_cpp_dir, args.model, args.quant, args.quant_out)
        payload["quantized_path"] = str(model_path)

    if args.prompt_file:
        prompt = args.prompt_file.read_text()
        payload["benchmark"] = bench_gguf(args.llama_cpp_dir, model_path, prompt, n_predict=args.n_predict)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
