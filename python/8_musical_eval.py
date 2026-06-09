#!/usr/bin/env python3
"""Run synthetic musical evaluation prompts against a saved NJam model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from super_njam.music_language import get_language
from super_njam.musical_eval import (
    DEFAULT_MAX_NEW_TOKENS,
    format_musical_eval_table,
    run_musical_eval,
    write_musical_eval_json,
)
from super_njam.training_tools import SentencePieceTokenizerAdapter, configure_torch_runtime


def _resolve_run_and_hf_dirs(model_dir: Path) -> tuple[Path, Path]:
    model_dir = model_dir.expanduser()
    if (model_dir / "config.json").exists() and (model_dir / "tokenizer.model").exists():
        return model_dir.parent, model_dir
    hf_dir = model_dir / "hf_model"
    if (hf_dir / "config.json").exists() and (hf_dir / "tokenizer.model").exists():
        return model_dir, hf_dir
    raise AssertionError(
        f"Expected --model-dir to point to an hf_model folder or a run folder containing hf_model: {model_dir}"
    )


def _detect_language(run_dir: Path, explicit_language: str) -> str:
    if explicit_language != "auto":
        return explicit_language
    summary_path = run_dir / "train_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            language = str(summary.get("language") or summary.get("config", {}).get("language") or "")
            if language:
                return language
        except Exception:
            pass
    return "njam-v3"


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run musical synthetic evaluations against a saved NJam model.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Run folder containing hf_model, or the hf_model folder itself.",
    )
    parser.add_argument(
        "--language",
        choices=["auto", "njam-v3", "njam-v2", "njam-v4"],
        default="auto",
        help="Language to use for synthetic prompts. auto reads train_summary.json when available.",
    )
    parser.add_argument(
        "--suite",
        choices=["light"],
        default="light",
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens generated for each synthetic prompt.",
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=4,
        help="Minimum recovered continuation notes required before harmony/rhythm scores are non-zero.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional JSON output path for detailed results.",
    )
    parser.add_argument(
        "--grammar-constrained-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply NJam-v4 grammar constraints during Python generation. Enabled by default for v4.",
    )
    args = parser.parse_args()

    configure_torch_runtime()
    run_dir, hf_dir = _resolve_run_and_hf_dirs(args.model_dir)
    language = get_language(_detect_language(run_dir, args.language))
    tokenizer = SentencePieceTokenizerAdapter(hf_dir / "tokenizer.model")
    device = _detect_device()
    model = LlamaForCausalLM.from_pretrained(str(hf_dir))
    model.to(device)
    result = run_musical_eval(
        model=model,
        tokenizer=tokenizer,
        language=language,
        suite=args.suite,
        max_new_tokens=args.max_new_tokens,
        min_notes=args.min_notes,
        device=device,
        grammar_constrained_generation=args.grammar_constrained_generation,
    )
    print(format_musical_eval_table(result))
    if args.json_out is not None:
        write_musical_eval_json(result, args.json_out)


if __name__ == "__main__":
    main()
