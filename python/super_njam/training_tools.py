"""Training utilities for llama-compatible NJam models."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lightning as L
import sentencepiece as spm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from .audio_tools import render_document_audio
from .midi_tools import write_midi
from .njam_v3 import (
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    analyze_parseable_continuation,
    encode_document,
    extract_header_metadata,
    parse_document,
    recover_continuation_document,
)

DEFAULT_TRAINING_SOUNDFONT = Path("soundfonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2")
MAX_SAMPLE_NOTE_SECONDS = 10.0


def load_corpus_records(corpus_path: Path) -> List[Dict[str, object]]:
    assert corpus_path.exists(), f"Corpus file does not exist: {corpus_path}"
    records = []
    for line in corpus_path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    assert records, f"No records found in corpus file: {corpus_path}"
    return records


def split_records_by_solo(
    records: Sequence[Dict[str, object]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[Dict[str, object]]]:
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1
    buckets = {"train": [], "val": [], "test": []}
    for record in records:
        melid = str(record["melid"]).encode("utf-8")
        bucket_value = int(hashlib.sha1(melid).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket_value < train_ratio:
            buckets["train"].append(record)
        elif bucket_value < train_ratio + val_ratio:
            buckets["val"].append(record)
        else:
            buckets["test"].append(record)
    assert buckets["train"] and buckets["val"] and buckets["test"], "Split produced an empty partition."
    return buckets


def build_sentencepiece_tokenizer(
    texts: Sequence[str],
    output_dir: Path,
    vocab_size: int = 2048,
) -> "SentencePieceTokenizerAdapter":
    assert texts, "build_sentencepiece_tokenizer requires non-empty texts."
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "sentencepiece_corpus.txt"
    corpus_path.write_text("\n".join(texts) + "\n")
    model_prefix = output_dir / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        hard_vocab_limit=False,
        max_sentence_length=1048576,
        byte_fallback=True,
    )
    return SentencePieceTokenizerAdapter(Path(str(model_prefix) + ".model"))


class SentencePieceTokenizerAdapter:
    def __init__(self, model_path: Path):
        assert model_path.exists(), f"SentencePiece model does not exist: {model_path}"
        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.bos_token_id = int(self.processor.bos_id())
        self.eos_token_id = int(self.processor.eos_id())
        self.unk_token_id = int(self.processor.unk_id())
        self.pad_token_id = self.eos_token_id
        self.vocab_size = int(self.processor.get_piece_size())

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        token_ids = [int(token_id) for token_id in ids]
        if skip_special_tokens:
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id not in {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            ]
        return str(self.processor.decode(token_ids))

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False) -> Dict[str, torch.Tensor]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        raise AssertionError(f"Unsupported return_tensors value: {return_tensors}")

    def save_pretrained(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        destination = out / "tokenizer.model"
        if self.model_path.resolve() != destination.resolve():
            shutil.copy2(self.model_path, destination)
        (out / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "SentencePieceTokenizerAdapter",
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                    "pad_token_id": self.pad_token_id,
                    "unk_token_id": self.unk_token_id,
                    "model_max_length": 1000000,
                },
                indent=2,
            )
            + "\n"
        )
        (out / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "</s>",
                },
                indent=2,
            )
            + "\n"
        )


def njam_body_text(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("NV3|"):
        return stripped
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 2:
        return "\n".join(lines[1:]).strip()
    body_match = re.search(r"\sT[-0-9A-Z]+", stripped)
    assert body_match is not None, "NJam text must contain body tokens after the header."
    return stripped[body_match.start() :].strip()


def njam_header_text(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("NV3|"):
        return ""
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        return lines[0]
    body_match = re.search(r"\sT[-0-9A-Z]+", stripped)
    assert body_match is not None, "NJam text must contain body tokens after the header."
    return stripped[: body_match.start()].strip()


def _prepare_solo_token_ids(
    text: str,
    tokenizer_model_path: str,
    bos_token_id: int,
    eos_token_id: int,
) -> Tuple[List[int], int]:
    processor = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
    body_text = njam_body_text(text)
    token_ids = [bos_token_id] + list(processor.encode(body_text, out_type=int)) + [eos_token_id]
    assert len(token_ids) >= 2, "Each solo must yield at least one next-token target."
    return token_ids, len(token_ids) - 1


def _build_dataset_executor(max_workers: int):
    try:
        return ProcessPoolExecutor(max_workers=max_workers), "process"
    except Exception:
        return ThreadPoolExecutor(max_workers=max_workers), "thread"


class SoloSlidingWindowDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: SentencePieceTokenizerAdapter,
        seq_len: int,
        split_name: str = "dataset",
        prep_workers: Optional[int] = None,
    ):
        assert texts, "SoloSlidingWindowDataset requires non-empty texts."
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        assert bos is not None and eos is not None, "Tokenizer must define BOS and EOS tokens."
        self.seq_len = seq_len
        self.pad_token_id = eos
        self.solo_token_ids: List[List[int]] = []
        self.windows: List[Tuple[int, int]] = []
        self.window_counts_per_solo: List[int] = []
        progress = (
            tqdm(total=len(texts), desc=f"Preparing {split_name} windows", unit="solo", leave=False, dynamic_ncols=True)
            if tqdm is not None
            else None
        )
        worker_count = 1 if prep_workers is None else min(len(texts), max(1, int(prep_workers)))
        if worker_count <= 1:
            for text in texts:
                token_ids, solo_window_count = _prepare_solo_token_ids(
                    text=text,
                    tokenizer_model_path=str(tokenizer.model_path),
                    bos_token_id=bos,
                    eos_token_id=eos,
                )
                solo_idx = len(self.solo_token_ids)
                self.solo_token_ids.append(token_ids)
                self.windows.extend((solo_idx, end_idx) for end_idx in range(solo_window_count))
                self.window_counts_per_solo.append(solo_window_count)
                if progress is not None:
                    progress.update(1)
        else:
            ordered_results: List[Optional[Tuple[List[int], int]]] = [None] * len(texts)
            executor, executor_kind = _build_dataset_executor(worker_count)
            print(f"Preparing {split_name} windows with {executor_kind} pool ({worker_count} workers)")
            with executor:
                future_to_index = {
                    executor.submit(
                        _prepare_solo_token_ids,
                        text,
                        str(tokenizer.model_path),
                        bos,
                        eos,
                    ): idx
                    for idx, text in enumerate(texts)
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    ordered_results[idx] = future.result()
                    if progress is not None:
                        progress.update(1)
            for result in ordered_results:
                assert result is not None
                token_ids, solo_window_count = result
                solo_idx = len(self.solo_token_ids)
                self.solo_token_ids.append(token_ids)
                self.windows.extend((solo_idx, end_idx) for end_idx in range(solo_window_count))
                self.window_counts_per_solo.append(solo_window_count)
        if progress is not None:
            progress.close()
        assert self.windows, "Sliding window construction produced zero samples."

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        solo_idx, end_idx = self.windows[idx]
        token_ids = self.solo_token_ids[solo_idx]
        chunk = token_ids[max(0, end_idx - self.seq_len + 1) : end_idx + 2]
        left_pad = (self.seq_len + 1) - len(chunk)
        padded = ([self.pad_token_id] * left_pad) + chunk
        input_ids = torch.tensor(padded[:-1], dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        if left_pad > 0:
            attention_mask[:left_pad] = 0
        labels = torch.tensor(padded[1:], dtype=torch.long)
        if left_pad > 0:
            labels[:left_pad] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class TrainConfig:
    corpus_path: Path
    output_dir: Path
    batch_size: int = 2
    seq_len: int = 256
    num_layers: int = 4
    hidden_size: int = 128
    num_heads: int = 4
    intermediate_size: int = 256
    max_epochs: int = 1
    learning_rate: float = 3e-4
    sample_prompt_ratio: float = 0.35
    sample_limit: int = 2
    sample_every_n_epochs: int = 1
    dataset_prep_workers: Optional[int] = None
    soundfont_path: Optional[Path] = DEFAULT_TRAINING_SOUNDFONT
    render_instrument: str = "saxophone"


class NJamLightningModule(L.LightningModule):
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: SentencePieceTokenizerAdapter,
        val_samples: Sequence[Dict[str, object]],
        config: TrainConfig,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.val_samples = list(val_samples)
        self.cfg = config
        self.save_hyperparameters(ignore=["model", "tokenizer", "val_samples"])

    def _sample_output_paths(self, epoch: int, sample_idx: int) -> Tuple[Path, Path, Path]:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.cfg.output_dir / f"sample_{epoch}_{sample_idx}"
        return (
            prefix.with_suffix(".njam"),
            prefix.with_suffix(".mid"),
            prefix.with_suffix(".wav"),
        )

    def _artifact_path(self, epoch: int, sample_idx: int, label: str, suffix: str) -> Path:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        return self.cfg.output_dir / f"sample_{epoch}_{sample_idx}.{label}.{suffix}"

    def _build_prompt(self, text: str) -> str:
        tokens = text.split()
        return " ".join(tokens[: max(8, int(len(tokens) * self.cfg.sample_prompt_ratio))])

    def _truncate_prompt_to_context_budget(self, prompt: str, reserved_new_tokens: int) -> str:
        max_positions = int(self.model.config.max_position_embeddings)
        max_prompt_tokens = max(1, max_positions - reserved_new_tokens - 1)
        if len(self.tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
            return prompt
        body_tokens = prompt.split()
        for start_idx in range(len(body_tokens)):
            candidate = " ".join(body_tokens[start_idx:]).strip()
            if candidate and len(self.tokenizer.encode(candidate, add_special_tokens=False)) <= max_prompt_tokens:
                return candidate
        return body_tokens[-1] if body_tokens else prompt.strip()

    def _generate_sample_text(self, prompt: str) -> Tuple[str, str, str]:
        max_positions = int(self.model.config.max_position_embeddings)
        target_new_tokens = min(64, max(1, max_positions - 2))
        effective_prompt = self._truncate_prompt_to_context_budget(prompt, target_new_tokens)
        encoded = self.tokenizer(effective_prompt, return_tensors="pt", add_special_tokens=False)
        encoded.pop("token_type_ids", None)
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        max_new_tokens = min(64, max(1, max_positions - input_length - 1))
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_k=16,
            )
        full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        continuation_ids = generated[0][input_length:]
        continuation_text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return effective_prompt, full_text, continuation_text

    def _write_sample_midi(self, document, midi_path: Path) -> None:
        render_doc = NJamDocument(
            metadata={**document.metadata, "render_instrument": self.cfg.render_instrument},
            events=list(document.events),
        )
        write_midi(render_doc, midi_path, max_note_seconds=MAX_SAMPLE_NOTE_SECONDS)

    def _sanitize_document_for_audio_render(self, document: NJamDocument) -> NJamDocument:
        ppq = int(document.metadata.get("ppq", "96"))
        max_time = ppq * 32
        tempo_bpm = float(document.metadata.get("tempo", "120.0"))
        max_duration = max(1, int(round(MAX_SAMPLE_NOTE_SECONDS * ppq * (tempo_bpm / 60.0))))
        sanitized_events = []
        for event in document.events:
            event_time = min(int(event.time), max_time)
            if isinstance(event, NoteEvent):
                sanitized_events.append(
                    NoteEvent(
                        time=event_time,
                        pitch=event.pitch,
                        velocity=event.velocity,
                        duration=max(1, min(int(event.duration), max_duration)),
                    )
                )
            elif isinstance(event, ControlChangeEvent):
                sanitized_events.append(
                    ControlChangeEvent(time=event_time, control=event.control, value=event.value)
                )
            elif isinstance(event, PitchBendEvent):
                sanitized_events.append(PitchBendEvent(time=event_time, value=event.value))
        return NJamDocument(metadata=dict(document.metadata), events=sanitized_events)

    def _write_sample_audio(self, document, wav_path: Path) -> bool:
        render_doc = NJamDocument(
            metadata={**document.metadata, "render_instrument": self.cfg.render_instrument},
            events=list(document.events),
        )
        try:
            render_document_audio(render_doc, wav_path, soundfont_path=self.cfg.soundfont_path)
        except Exception:
            try:
                sanitized_doc = self._sanitize_document_for_audio_render(render_doc)
                render_document_audio(sanitized_doc, wav_path, soundfont_path=self.cfg.soundfont_path)
            except Exception:
                if wav_path.exists():
                    wav_path.unlink()
                return False
        if not wav_path.exists() or wav_path.stat().st_size <= 44:
            if wav_path.exists():
                wav_path.unlink()
            return False
        return True

    def _write_render_bundle(self, document, epoch: int, sample_idx: int, label: str, text_out: str) -> Dict[str, str]:
        njam_path = self._artifact_path(epoch, sample_idx, label, "njam")
        midi_path = self._artifact_path(epoch, sample_idx, label, "mid")
        wav_path = self._artifact_path(epoch, sample_idx, label, "wav")
        njam_path.write_text(text_out, encoding="utf-8")
        self._write_sample_midi(document, midi_path)
        paths = {
            "njam": str(njam_path),
            "midi": str(midi_path),
        }
        if self._write_sample_audio(document, wav_path):
            paths["wav"] = str(wav_path)
        return paths

    def _log_sample_audio(self, wav_path: Path, sample_idx: int) -> None:
        logger = self.logger.experiment if self.logger else None
        if logger is None or not wav_path.exists():
            return
        try:
            audio_tensor = torch.tensor(_read_wav_mono(wav_path), dtype=torch.float32).unsqueeze(0)
        except Exception:
            return
        logger.add_audio(f"samples_audio/sample_{sample_idx}", audio_tensor, self.current_epoch, sample_rate=22050)

    def _log_sample_text(self, sample_idx: int, text_out: str, model_only_text: str) -> None:
        logger = self.logger.experiment if self.logger else None
        if logger is not None:
            logger.add_text(f"samples/generated_full_{sample_idx}", text_out, self.current_epoch)
            logger.add_text(f"samples/generated_model_only_{sample_idx}", model_only_text, self.current_epoch)

    def _log_sample_error(self, sample_idx: int, exc: Exception) -> None:
        logger = self.logger.experiment if self.logger else None
        if logger is not None:
            logger.add_text(f"samples/sample_{sample_idx}_error", str(exc), self.current_epoch)

    def _log_sample_metrics(self, sample_idx: int, stats: Dict[str, float | int]) -> None:
        logger = self.logger.experiment if self.logger else None
        if logger is not None:
            for key, value in stats.items():
                logger.add_scalar(f"samples_metrics/generated_model_only_{key}_{sample_idx}", value, self.current_epoch)

    def _write_sample_summary(self, sample_idx: int, payload: Dict[str, object]) -> None:
        summary_path = self._artifact_path(self.current_epoch, sample_idx, "summary", "json")
        summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    def _remove_render_bundle(self, paths: Dict[str, str]) -> None:
        for path_str in paths.values():
            path = Path(path_str)
            if path.exists():
                path.unlink()

    def _slice_model_only_document(self, generated_doc: NJamDocument, prompt: str) -> Optional[NJamDocument]:
        prompt_doc = parse_document(prompt)
        prompt_event_count = len(prompt_doc.sorted_events())
        generated_events = generated_doc.sorted_events()
        if len(generated_events) <= prompt_event_count:
            return None
        sliced_events = generated_events[prompt_event_count:]
        start_time = sliced_events[0].time
        rebased_events = []
        for event in sliced_events:
            rebased_time = event.time - start_time
            rebased_events.append(event.__class__(**{**event.__dict__, "time": rebased_time}))
        return NJamDocument(metadata=dict(generated_doc.metadata), events=rebased_events)

    def _recover_model_only_document(self, prompt: str, model_only_text: str) -> Optional[NJamDocument]:
        metadata = extract_header_metadata(prompt)
        return recover_continuation_document(model_only_text, metadata=metadata)

    def _write_model_only_render_bundle(
        self,
        sample_idx: int,
        document: NJamDocument,
        summary: Dict[str, object],
        render_mode: str,
    ) -> None:
        model_only_paths = self._write_render_bundle(
            document,
            self.current_epoch,
            sample_idx,
            "generated_model_only",
            encode_document(document),
        )
        summary["generated_model_only_parse_ok"] = True
        summary["generated_model_only_paths"] = model_only_paths
        summary["generated_model_only_render_mode"] = render_mode
        if "wav" in model_only_paths:
            self._log_sample_audio(Path(model_only_paths["wav"]), sample_idx)

    def _write_reference_once(self, sample_idx: int, text: str, summary: Dict[str, object]) -> None:
        if self.current_epoch != 0:
            summary["reference_paths"] = "written_at_epoch_0_only"
            return
        try:
            reference_doc = parse_document(text)
            summary["reference_paths"] = self._write_render_bundle(
                reference_doc,
                self.current_epoch,
                sample_idx,
                "reference",
                text,
            )
        except Exception as exc:
            summary["reference_error"] = str(exc)

    def _render_validation_sample(self, sample_idx: int, sample: Dict[str, object]) -> None:
        text = str(sample["text"])
        body_text = njam_body_text(text)
        prompt = self._build_prompt(body_text)
        effective_prompt, full_body_text, model_only_text = self._generate_sample_text(prompt)
        self._log_sample_text(sample_idx, full_body_text, model_only_text)
        model_only_recovery_stats = analyze_parseable_continuation(model_only_text).to_dict()
        self._log_sample_metrics(sample_idx, model_only_recovery_stats)
        header_text = njam_header_text(text)
        generated_text = header_text + "\n" + full_body_text.strip() + "\n"
        summary: Dict[str, object] = {
            "epoch": int(self.current_epoch),
            "sample_idx": int(sample_idx),
            "prompt": effective_prompt,
            "generated_text_preview": full_body_text[:500],
            "generated_model_only_preview": model_only_text[:500],
            "generated_model_only_recovery_stats": model_only_recovery_stats,
            "generated_parse_ok": False,
            "generated_model_only_parse_ok": False,
        }
        try:
            generated_doc = parse_document(generated_text)
            summary["generated_parse_ok"] = True
            prompt_text = header_text + "\n" + effective_prompt.strip() + "\n"
            model_only_doc = self._slice_model_only_document(generated_doc, prompt_text)
            if model_only_doc is not None and model_only_doc.events:
                generated_paths = self._write_render_bundle(
                    generated_doc,
                    self.current_epoch,
                    sample_idx,
                    "generated_full",
                    generated_text,
                )
                summary["generated_full_paths"] = generated_paths
                self._write_model_only_render_bundle(sample_idx, model_only_doc, summary, render_mode="strict")
            else:
                recovered_model_only_doc = self._recover_model_only_document(text, model_only_text)
                if recovered_model_only_doc is not None and recovered_model_only_doc.events:
                    self._write_model_only_render_bundle(sample_idx, recovered_model_only_doc, summary, render_mode="recovered")
                else:
                    summary["generated_error"] = "Model output parsed, but no standalone parseable model-only continuation remained after trimming the prompt."
        except Exception as exc:
            recovered_model_only_doc = self._recover_model_only_document(text, model_only_text)
            if recovered_model_only_doc is not None and recovered_model_only_doc.events:
                self._write_model_only_render_bundle(sample_idx, recovered_model_only_doc, summary, render_mode="recovered")
                summary["generated_error"] = str(exc)
            else:
                summary["generated_error"] = str(exc)
                self._log_sample_error(sample_idx, exc)
        self._write_reference_once(sample_idx, text, summary)
        self._write_sample_summary(sample_idx, summary)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, labels=labels).loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        perplexity = torch.exp(loss.detach())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.val_samples:
            return
        assert self.cfg.sample_every_n_epochs >= 1, "sample_every_n_epochs must be at least 1."
        if self.current_epoch % self.cfg.sample_every_n_epochs != 0:
            return
        for idx, sample in enumerate(self.val_samples[: self.cfg.sample_limit]):
            self._render_validation_sample(idx, sample)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)


def _read_wav_mono(path: Path) -> List[float]:
    import wave

    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        data = torch.frombuffer(frames, dtype=torch.int16).float() / 32767.0
    return data.tolist()


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True


def detect_accelerator() -> Dict[str, object]:
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1, "precision": "16-mixed"}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1, "precision": "32-true"}
    return {"accelerator": "cpu", "devices": 1, "precision": "32-true"}


def dataloader_kwargs() -> Dict[str, object]:
    if torch.cuda.is_available():
        num_workers = min(4, os.cpu_count() or 1)
        return {
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
    return {"num_workers": 0, "pin_memory": False}


def run_training(config: TrainConfig) -> Dict[str, object]:
    configure_torch_runtime()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_corpus_records(config.corpus_path)
    splits = split_records_by_solo(records)
    tokenizer_dir = config.output_dir / "tokenizer"
    tokenizer = build_sentencepiece_tokenizer([njam_body_text(str(record["text"])) for record in splits["train"]], tokenizer_dir)
    tokenizer.save_pretrained(str(tokenizer_dir))
    train_ds = SoloSlidingWindowDataset(
        [str(r["text"]) for r in splits["train"]],
        tokenizer,
        config.seq_len,
        split_name="train",
        prep_workers=config.dataset_prep_workers,
    )
    val_ds = SoloSlidingWindowDataset(
        [str(r["text"]) for r in splits["val"]],
        tokenizer,
        config.seq_len,
        split_name="val",
        prep_workers=config.dataset_prep_workers,
    )
    test_ds = SoloSlidingWindowDataset(
        [str(r["text"]) for r in splits["test"]],
        tokenizer,
        config.seq_len,
        split_name="test",
        prep_workers=config.dataset_prep_workers,
    )

    model_cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_heads,
        max_position_embeddings=config.seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(model_cfg)
    module = NJamLightningModule(model=model, tokenizer=tokenizer, val_samples=splits["val"], config=config)
    logger = TensorBoardLogger(save_dir=str(config.output_dir), name="tensorboard")
    checkpoint = ModelCheckpoint(
        dirpath=str(config.output_dir / "checkpoints"),
        filename="best",
        save_top_k=1,
        save_last=False,
        monitor="val_loss",
        mode="min",
        save_on_train_epoch_end=False,
    )
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[checkpoint],
        enable_checkpointing=True,
        log_every_n_steps=1,
        **detect_accelerator(),
    )
    trainer.fit(
        module,
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **dataloader_kwargs()),
        DataLoader(val_ds, batch_size=config.batch_size, **dataloader_kwargs()),
    )
    hf_dir = config.output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))
    summary = {
        "best_model_path": checkpoint.best_model_path,
        "hf_model_dir": str(hf_dir),
        "dataset_mode": "solo_sliding",
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "mean_train_windows_per_solo": (sum(train_ds.window_counts_per_solo) / len(train_ds.window_counts_per_solo)),
        "mean_val_windows_per_solo": (sum(val_ds.window_counts_per_solo) / len(val_ds.window_counts_per_solo)),
        "mean_test_windows_per_solo": (sum(test_ds.window_counts_per_solo) / len(test_ds.window_counts_per_solo)),
        "window_stride": 1,
        "header_tokens_dropped": True,
        "left_padding": True,
        "pad_loss_masked": True,
        "dataset_prep_workers": config.dataset_prep_workers,
        "tokenizer_dir": str(config.output_dir / "tokenizer"),
        "config": asdict(config),
    }
    (config.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def run_structured_sweep(base_config: TrainConfig, output_path: Path) -> List[Dict[str, object]]:
    sweep = [
        {"num_layers": 2, "hidden_size": 96, "num_heads": 4, "intermediate_size": 192},
        {"num_layers": 4, "hidden_size": 128, "num_heads": 4, "intermediate_size": 256},
        {"num_layers": 6, "hidden_size": 192, "num_heads": 6, "intermediate_size": 384},
    ]
    results = []
    for idx, override in enumerate(sweep):
        run_dir = base_config.output_dir / f"run_{idx:02d}"
        cfg = TrainConfig(**{**asdict(base_config), **override, "output_dir": run_dir})
        result = run_training(cfg)
        results.append(result)
    output_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    return results
