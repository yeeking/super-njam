"""GGUF export and llama.cpp benchmark helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExportConfig:
    model_dir: Path
    output_dir: Path
    llama_cpp_dir: Path
    outfile: str = "model-f16.gguf"
    outtype: str = "f16"


def _run(command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    assert result.returncode == 0, (
        f"Command failed with code {result.returncode}: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def export_hf_to_gguf(config: ExportConfig) -> Path:
    assert config.model_dir.exists(), f"Model directory does not exist: {config.model_dir}"
    convert_script = config.llama_cpp_dir / "convert_hf_to_gguf.py"
    assert convert_script.exists(), f"llama.cpp convert script not found: {convert_script}"
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
