"""GGUF export and llama.cpp benchmark helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .training_tools import SentencePieceTokenizerAdapter


@dataclass
class ExportConfig:
    model_dir: Path
    output_dir: Path
    llama_cpp_dir: Path
    outfile: str = "model-f16.gguf"
    outtype: str = "f16"


@dataclass
class CheckpointExportConfig:
    run_dir: Path
    checkpoint_path: Path
    output_dir: Path
    llama_cpp_dir: Path
    outfile: str = "model-f16.gguf"
    outtype: str = "f16"
    temp_model_dirname: str = "hf_model_from_ckpt"


def _run(command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    assert result.returncode == 0, (
        f"Command failed with code {result.returncode}: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def assert_llama_cpp_checkout(llama_cpp_dir: Path) -> None:
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if convert_script.exists():
        return
    libs_readme = Path("libs/README.md")
    suggestion = (
        "llama.cpp checkout not found. Expected to find "
        f"{convert_script}. Clone llama.cpp into libs/ first."
    )
    if libs_readme.exists():
        suggestion += (
            f" See {libs_readme} for the local clone instruction, "
            "for example: git clone git@github.com:ggml-org/llama.cpp.git libs/llama.cpp"
        )
    raise AssertionError(suggestion)


def export_checkpoint_to_hf_model(run_dir: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    assert run_dir.exists(), f"Run directory does not exist: {run_dir}"
    assert checkpoint_path.exists(), f"Checkpoint does not exist: {checkpoint_path}"
    summary_path = run_dir / "train_summary.json"
    assert summary_path.exists(), f"Run summary not found: {summary_path}"
    tokenizer_model = run_dir / "tokenizer" / "tokenizer.model"
    assert tokenizer_model.exists(), f"Tokenizer model not found: {tokenizer_model}"

    summary = json.loads(summary_path.read_text())
    cfg = summary["config"]
    tokenizer = SentencePieceTokenizerAdapter(tokenizer_model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    model_cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_layers"]),
        num_attention_heads=int(cfg["num_heads"]),
        num_key_value_heads=int(cfg["num_heads"]),
        max_position_embeddings=int(cfg["seq_len"]),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(model_cfg)
    model_weights = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(model_weights, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def export_hf_to_gguf(config: ExportConfig) -> Path:
    assert config.model_dir.exists(), f"Model directory does not exist: {config.model_dir}"
    assert_llama_cpp_checkout(config.llama_cpp_dir)
    convert_script = config.llama_cpp_dir / "convert_hf_to_gguf.py"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / config.outfile
    _run(
        [
            sys.executable,
            str(convert_script),
            str(config.model_dir),
            "--outfile",
            str(output_path),
            "--outtype",
            config.outtype,
        ]
    )
    return output_path


def export_checkpoint_to_gguf(config: CheckpointExportConfig) -> Path:
    temp_model_dir = config.run_dir / config.temp_model_dirname
    export_checkpoint_to_hf_model(config.run_dir, config.checkpoint_path, temp_model_dir)
    return export_hf_to_gguf(
        ExportConfig(
            model_dir=temp_model_dir,
            output_dir=config.output_dir,
            llama_cpp_dir=config.llama_cpp_dir,
            outfile=config.outfile,
            outtype=config.outtype,
        )
    )


def quantize_gguf(llama_cpp_dir: Path, source_path: Path, quantization: str, output_path: Path) -> Path:
    binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    assert binary.exists(), f"Quantize binary not found: {binary}"
    _run([str(binary), str(source_path), str(output_path), quantization])
    return output_path


def bench_gguf(
    llama_cpp_dir: Path,
    model_path: Path,
    prompt: str,
    n_predict: int = 64,
) -> Dict[str, object]:
    binary = llama_cpp_dir / "build" / "bin" / "llama-bench"
    assert binary.exists(), f"llama-bench binary not found: {binary}"
    result = _run([str(binary), "-m", str(model_path), "-p", prompt, "-n", str(n_predict)])
    return {"stdout": result.stdout, "stderr": result.stderr, "model_path": str(model_path)}
