"""Training utilities for llama-compatible NJam models."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import re
import shutil
import warnings
from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lightning as L
import sentencepiece as spm
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from .audio_tools import render_document_audio
from .generation_tools import build_generation_logits_processor
from .midi_tools import write_midi
from .music_language import get_language
from .musical_eval import (
    DEFAULT_MAX_NEW_TOKENS as DEFAULT_MUSICAL_EVAL_MAX_NEW_TOKENS,
    log_musical_eval_to_tensorboard,
    run_musical_eval,
    write_musical_eval_json,
)
from .njam_v3 import (
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
)

DEFAULT_TRAINING_SOUNDFONT = Path("soundfonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2")
MAX_SAMPLE_NOTE_SECONDS = 10.0
MAX_TRAINING_CHARS = 60_000

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
    if not (buckets["train"] and buckets["val"] and buckets["test"]):
        assert len(records) >= 3, "At least 3 solos are required to create train/val/test splits."
        sorted_records = sorted(
            records,
            key=lambda record: hashlib.sha1(str(record["melid"]).encode("utf-8")).hexdigest(),
        )
        buckets = {
            "train": list(sorted_records[:-2]),
            "val": [sorted_records[-2]],
            "test": [sorted_records[-1]],
        }
    return buckets


def build_sentencepiece_tokenizer(
    texts: Sequence[str],
    output_dir: Path,
    vocab_size: int = 16000,# allows for a nice and clever tokenizer 
    model_type: str = "unigram",
    trainer_kwargs: Optional[Dict[str, object]] = None,
) -> "SentencePieceTokenizerAdapter":
    assert texts, "build_sentencepiece_tokenizer requires non-empty texts."
    MAX_LEN = 50000

    ## simple way to avoid long text lines that crash the tokenizer trainer: 
    # for ind,t in enumerate(texts):
    #     if len(t) > MAX_LEN: texts[ind] = t[0:MAX_LEN]

    ## complex way to avoid long texts crashing the e coder 
    # retain all text but insert line breaks to long ones
    for ind,t in enumerate(texts):
        if len(t) > MAX_LEN: 
            texts[ind] = "\n".join(
                t[i:i + MAX_LEN]
                for i in range(0, len(t), MAX_LEN)
            )

    ## write the texts as a single corpus file

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "sentencepiece_corpus.txt"
    corpus_path.write_text("\n".join(texts) + "\n")

    ## write the texts to individual files in the corpus path folder
    # corpus_files = []
    # corpus_path = output_dir / "sp_training_corpus"
    # corpus_path.mkdir(parents=True, exist_ok=True)
    # for ind,t in enumerate(texts):
    #     fname = corpus_path / f"{ind}.txt"
    #     fname.write_text(t)
    #     corpus_files.append(str(fname))
    # corpus_files = ",".join(corpus_files)
    
    model_prefix = output_dir / "tokenizer"
    spm.set_min_log_level(2)
    resolved_trainer_kwargs = dict(trainer_kwargs or {})
    byte_fallback = bool(resolved_trainer_kwargs.pop("byte_fallback", True))
    print(f"[LOG] calling train on tokenizer. using corpus of {len(texts)} texts vocab size {vocab_size} from corpus {corpus_path}")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        # input=corpus_files, 
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        hard_vocab_limit=False,
        # max_sentence_length=1048576,# commenting out as seems arbitrary 
        byte_fallback=byte_fallback,
        **resolved_trainer_kwargs,
    )
    tokenizer = SentencePieceTokenizerAdapter(Path(str(model_prefix) + ".model"))
    tokenizer.tokenizer_model_type = model_type
    return tokenizer


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
        self.loss_mask_token_ids: set[int] = set()
        self.tokenizer_model_type: str = "sentencepiece"
        config_path = model_path.parent / "tokenizer_config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                self.loss_mask_token_ids = {int(token_id) for token_id in config.get("loss_mask_token_ids", [])}
                self.tokenizer_model_type = str(config.get("tokenizer_model_type", self.tokenizer_model_type))
            except Exception:
                self.loss_mask_token_ids = set()

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
                    "loss_mask_token_ids": sorted(int(token_id) for token_id in getattr(self, "loss_mask_token_ids", set())),
                    "tokenizer_model_type": getattr(self, "tokenizer_model_type", "sentencepiece"),
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


def njam_body_text(text: str, language: str = "njam-v3") -> str:
    return get_language(language).body_text(text)


def njam_header_text(text: str, language: str = "njam-v3") -> str:
    return get_language(language).header_text(text)


def _prepare_solo_token_ids(
    text: str,
    tokenizer_model_path: str,
    bos_token_id: int,
    eos_token_id: int,
    language: str,
) -> Tuple[List[int], int]:
    processor = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
    body_text = njam_body_text(text, language=language)
    token_ids = [bos_token_id] + list(processor.encode(body_text, out_type=int)) + [eos_token_id]
    assert len(token_ids) >= 2, "Each solo must yield at least one next-token target."
    return token_ids, len(token_ids) - 1


def _build_dataset_executor(max_workers: int):
    try:
        return ProcessPoolExecutor(max_workers=max_workers), "process"
    except Exception:
        return ThreadPoolExecutor(max_workers=max_workers), "thread"


def _load_solo_token_ids(
    texts: Sequence[str],
    tokenizer: SentencePieceTokenizerAdapter,
    split_name: str,
    prep_workers: Optional[int],
    language: str,
) -> Tuple[List[List[int]], List[int]]:
    assert texts, "Sliding-window datasets require non-empty texts."
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    assert bos is not None and eos is not None, "Tokenizer must define BOS and EOS tokens."
    solo_token_ids: List[List[int]] = []
    window_counts_per_solo: List[int] = []
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
                language=language,
            )
            solo_token_ids.append(token_ids)
            window_counts_per_solo.append(solo_window_count)
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
                    language,
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
            solo_token_ids.append(token_ids)
            window_counts_per_solo.append(solo_window_count)
    if progress is not None:
        progress.close()
    assert sum(window_counts_per_solo) > 0, "Sliding window construction produced zero samples."
    return solo_token_ids, window_counts_per_solo


def _window_sample(
    token_ids: List[int],
    end_idx: int,
    seq_len: int,
    pad_token_id: int,
    loss_mask_token_ids: Optional[set[int]] = None,
) -> Dict[str, torch.Tensor]:
    chunk = token_ids[max(0, end_idx - seq_len + 1) : end_idx + 2]
    left_pad = (seq_len + 1) - len(chunk)
    padded = ([pad_token_id] * left_pad) + chunk
    input_ids = torch.tensor(padded[:-1], dtype=torch.long)
    attention_mask = torch.ones(seq_len, dtype=torch.long)
    if left_pad > 0:
        attention_mask[:left_pad] = 0
    labels = torch.tensor(padded[1:], dtype=torch.long)
    if left_pad > 0:
        labels[:left_pad] = -100
    if loss_mask_token_ids:
        for token_id in loss_mask_token_ids:
            labels[labels == int(token_id)] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SoloSlidingWindowDataset(Dataset):
    """Provide data as follows:
    yield all possible windows of length seq_length on all texts
    """
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: SentencePieceTokenizerAdapter,
        seq_len: int,
        split_name: str = "dataset",
        prep_workers: Optional[int] = None,
        language: str = "njam-v3",
    ):
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.eos_token_id
        self.loss_mask_token_ids = set(getattr(tokenizer, "loss_mask_token_ids", set()))
        self.solo_token_ids, self.window_counts_per_solo = _load_solo_token_ids(
            texts=texts,
            tokenizer=tokenizer,
            split_name=split_name,
            prep_workers=prep_workers,
            language=language,
        )
        self.windows: List[Tuple[int, int]] = []
        for solo_idx, solo_window_count in enumerate(self.window_counts_per_solo):
            self.windows.extend((solo_idx, end_idx) for end_idx in range(solo_window_count))
        assert self.windows, "Sliding window construction produced zero samples."

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        solo_idx, end_idx = self.windows[idx]
        return _window_sample(
            self.solo_token_ids[solo_idx],
            end_idx,
            self.seq_len,
            self.pad_token_id,
            self.loss_mask_token_ids,
        )


class SoloSlidingWindowDatasetPartial(Dataset):
    """provides data according to the following pattern:
    in one epoch, per text (song) provide batch_size sliding windows.
    on the next epoch, same per text, but with an offset which is where we left off
    
    """
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: SentencePieceTokenizerAdapter,
        seq_len: int,
        batch_size: int,
        split_name: str = "dataset",
        prep_workers: Optional[int] = None,
        randomize_each_epoch: bool = False,
        language: str = "njam-v3",
    ):
        assert batch_size >= 1, "SoloSlidingWindowDatasetPartial requires batch_size >= 1."
        self.seq_len = seq_len
        self.batch_size = int(batch_size)
        self.pad_token_id = tokenizer.eos_token_id
        self.loss_mask_token_ids = set(getattr(tokenizer, "loss_mask_token_ids", set()))
        self.randomize_each_epoch = randomize_each_epoch
        self.solo_token_ids, self.window_counts_per_solo = _load_solo_token_ids(
            texts=texts,
            tokenizer=tokenizer,
            split_name=split_name,
            prep_workers=prep_workers,
            language=language,
        )
        self.window_cursors = [0 for _ in self.window_counts_per_solo]

    def __len__(self) -> int:
        return len(self.solo_token_ids) * self.batch_size

    def resolve_window_index(self, idx: int) -> Tuple[int, int]:
        solo_idx = idx // self.batch_size
        offset = idx % self.batch_size
        window_count = self.window_counts_per_solo[solo_idx]
        return solo_idx, (self.window_cursors[solo_idx] + offset) % window_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        solo_idx, end_idx = self.resolve_window_index(idx)
        return _window_sample(
            self.solo_token_ids[solo_idx],
            end_idx,
            self.seq_len,
            self.pad_token_id,
            self.loss_mask_token_ids,
        )

    def advance_epoch(self) -> None:
        for solo_idx, window_count in enumerate(self.window_counts_per_solo):
            self.window_cursors[solo_idx] = (self.window_cursors[solo_idx] + self.batch_size) % window_count

    def reset_cursors(self) -> None:
        self.window_cursors = [0 for _ in self.window_counts_per_solo]

    def prepare_validation_epoch(self) -> None:
        if self.randomize_each_epoch:
            self.window_cursors = [random.randrange(window_count) for window_count in self.window_counts_per_solo]

    def finish_validation_epoch(self) -> None:
        if not self.randomize_each_epoch:
            self.advance_epoch()


def _parse_signature_ticks(metadata: Dict[str, str]) -> tuple[int, int, int]:
    """Resolve PPQ, time-signature numerator, and bar length in ticks.

    Args:
        metadata: NJam document metadata, usually containing ``ppq`` and ``sig``.
    """
    ppq = int(metadata.get("ppq", "96"))
    sig = str(metadata.get("sig", "4/4"))
    try:
        numerator_raw, denominator_raw = sig.split("/", 1)
        numerator = max(1, int(numerator_raw))
        denominator = max(1, int(denominator_raw))
    except Exception:
        numerator, denominator = 4, 4
    bar_ticks = max(1, int(round(ppq * numerator * (4.0 / denominator))))
    return ppq, numerator, bar_ticks


def _rebased_event(event, start_time: int):
    payload = {**event.__dict__, "time": max(0, int(event.time) - int(start_time))}
    return event.__class__(**payload)


def _event_is_note(event) -> bool:
    return isinstance(event, NoteEvent)


def _tokenize_musical_window(
    document: NJamDocument,
    language,
    tokenizer: SentencePieceTokenizerAdapter,
) -> List[int]:
    text = language.encode_document(document)
    body = language.body_text(text)
    return [tokenizer.bos_token_id] + tokenizer.encode(body, add_special_tokens=False) + [tokenizer.eos_token_id]


class SoloMusicalWindowDatasetPartial(Dataset):
    """Partial dataset over musically aligned bar windows.

    Each dataset item is one normal next-token sample. A batch contains multiple
    candidate windows from one source text, and an epoch yields one batch per
    retained source text.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: SentencePieceTokenizerAdapter,
        seq_len: int,
        batch_size: int,
        split_name: str = "dataset",
        randomize_each_epoch: bool = False,
        language: str = "njam-v3",
        musical_window_bars: int = 4,
        musical_window_hop_bars: int = 1,
        min_window_notes: int = 8,
    ):
        assert batch_size >= 1, "SoloMusicalWindowDatasetPartial requires batch_size >= 1."
        assert musical_window_bars >= 1, "musical_window_bars must be at least 1."
        assert musical_window_hop_bars >= 1, "musical_window_hop_bars must be at least 1."
        assert min_window_notes >= 1, "min_window_notes must be at least 1."
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.pad_token_id = tokenizer.eos_token_id
        self.loss_mask_token_ids = set(getattr(tokenizer, "loss_mask_token_ids", set()))
        self.randomize_each_epoch = randomize_each_epoch
        self.language = get_language(language)
        self.musical_window_bars = int(musical_window_bars)
        self.musical_window_hop_bars = int(musical_window_hop_bars)
        self.min_window_notes = int(min_window_notes)
        self.solo_token_ids: List[List[List[int]]] = []
        self.window_counts_per_solo: List[int] = []
        self.build_stats: Dict[str, int | str] = {
            "mode": split_name,
            "input_texts": len(texts),
            "retained_texts": 0,
            "excluded_texts": 0,
            "candidate_windows": 0,
            "sparse_windows_skipped": 0,
            "overlength_windows_skipped": 0,
            "parse_failures": 0,
            "fallback_windows": 0,
        }
        for text in texts:
            try:
                candidates = self._build_candidates_for_text(str(text), tokenizer)
            except Exception:
                self.build_stats["parse_failures"] = int(self.build_stats["parse_failures"]) + 1
                candidates = []
            if candidates:
                self.solo_token_ids.append(candidates)
                self.window_counts_per_solo.append(len(candidates))
                self.build_stats["retained_texts"] = int(self.build_stats["retained_texts"]) + 1
                self.build_stats["candidate_windows"] = int(self.build_stats["candidate_windows"]) + len(candidates)
            else:
                self.build_stats["excluded_texts"] = int(self.build_stats["excluded_texts"]) + 1
        assert self.solo_token_ids, (
            f"{split_name} musical-window dataset produced zero usable solos. "
            "Try lowering --min-window-notes or increasing --seq-len."
        )
        self.window_cursors = [0 for _ in self.window_counts_per_solo]

    def _subdocument_for_window(
        self,
        document: NJamDocument,
        start_time: int,
        end_time: int,
        min_notes: int,
    ) -> NJamDocument | None:
        events = [
            _rebased_event(event, start_time)
            for event in document.sorted_events()
            if int(start_time) <= int(event.time) < int(end_time)
        ]
        if sum(1 for event in events if _event_is_note(event)) < min_notes:
            return None
        metadata = {key: value for key, value in document.metadata.items() if not str(key).startswith("_")}
        return NJamDocument(metadata=metadata, events=events)

    def _candidate_fallback(self, document: NJamDocument, tokenizer: SentencePieceTokenizerAdapter) -> List[int] | None:
        notes = [event for event in document.sorted_events() if _event_is_note(event)]
        if not notes:
            return None
        _, _, bar_ticks = _parse_signature_ticks(document.metadata)
        window_ticks = max(1, self.musical_window_bars * bar_ticks)
        best: List[int] | None = None
        for note in notes:
            subdoc = self._subdocument_for_window(document, int(note.time), int(note.time) + window_ticks, min_notes=1)
            if subdoc is None:
                continue
            token_ids = _tokenize_musical_window(subdoc, self.language, tokenizer)
            if len(token_ids) <= self.seq_len + 1 and (best is None or len(token_ids) < len(best)):
                best = token_ids
        if best is not None:
            self.build_stats["fallback_windows"] = int(self.build_stats["fallback_windows"]) + 1
        return best

    def _build_candidates_for_text(self, text: str, tokenizer: SentencePieceTokenizerAdapter) -> List[List[int]]:
        """Build tokenized bar-window candidates for one encoded solo.

        Args:
            text: Full NJam document text.
            tokenizer: Run tokenizer used to turn each rebased subdocument body
                into token ids.
        """
        document = self.language.parse_document(text)
        sorted_events = document.sorted_events()
        if not sorted_events:
            return []
        _, _, bar_ticks = _parse_signature_ticks(document.metadata)
        window_ticks = max(1, self.musical_window_bars * bar_ticks)
        hop_ticks = max(1, self.musical_window_hop_bars * bar_ticks)
        max_time = max(int(event.time) for event in sorted_events)
        candidates: List[List[int]] = []
        for start_time in range(0, max_time + 1, hop_ticks):
            subdoc = self._subdocument_for_window(
                document,
                start_time,
                start_time + window_ticks,
                min_notes=self.min_window_notes,
            )
            if subdoc is None:
                self.build_stats["sparse_windows_skipped"] = int(self.build_stats["sparse_windows_skipped"]) + 1
                continue
            token_ids = _tokenize_musical_window(subdoc, self.language, tokenizer)
            if len(token_ids) > self.seq_len + 1:
                self.build_stats["overlength_windows_skipped"] = int(self.build_stats["overlength_windows_skipped"]) + 1
                continue
            candidates.append(token_ids)
        if not candidates:
            fallback = self._candidate_fallback(document, tokenizer)
            if fallback is not None:
                candidates.append(fallback)
        return candidates

    def __len__(self) -> int:
        return len(self.solo_token_ids) * self.batch_size

    def resolve_window_index(self, idx: int) -> Tuple[int, int]:
        solo_idx = idx // self.batch_size
        offset = idx % self.batch_size
        window_count = self.window_counts_per_solo[solo_idx]
        return solo_idx, (self.window_cursors[solo_idx] + offset) % window_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        solo_idx, candidate_idx = self.resolve_window_index(idx)
        token_ids = self.solo_token_ids[solo_idx][candidate_idx]
        return _window_sample(
            token_ids,
            len(token_ids) - 2,
            self.seq_len,
            self.pad_token_id,
            self.loss_mask_token_ids,
        )

    def advance_epoch(self) -> None:
        for solo_idx, window_count in enumerate(self.window_counts_per_solo):
            self.window_cursors[solo_idx] = (self.window_cursors[solo_idx] + self.batch_size) % window_count

    def reset_cursors(self) -> None:
        self.window_cursors = [0 for _ in self.window_counts_per_solo]

    def prepare_validation_epoch(self) -> None:
        if self.randomize_each_epoch:
            self.window_cursors = [random.randrange(window_count) for window_count in self.window_counts_per_solo]

    def finish_validation_epoch(self) -> None:
        if not self.randomize_each_epoch:
            self.advance_epoch()


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
    sample_every_n_items: Optional[int] = None
    dataset_prep_workers: Optional[int] = None
    soundfont_path: Optional[Path] = DEFAULT_TRAINING_SOUNDFONT
    render_instrument: str = "saxophone"
    validation_preflight: bool = False
    validation_preflight_val_batches: int = 1
    early_stopping_patience: int = 3
    dataset_mode: str = "partial"
    validation_dataset_mode: str = "partial-random"
    language: str = "njam-v3"
    musical_eval: bool = True
    musical_eval_every_n_epochs: int = 1
    musical_eval_max_new_tokens: int = DEFAULT_MUSICAL_EVAL_MAX_NEW_TOKENS
    grammar_constrained_generation: bool = True
    musical_window_bars: int = 4
    musical_window_hop_bars: int = 1
    min_window_notes: int = 8
    gradient_accumulation_steps: int = 16
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"


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
        self.language = get_language(config.language)
        self._items_seen_for_sample_render = 0
        self._next_sample_render_item_target = (
            None if config.sample_every_n_items is None else int(config.sample_every_n_items)
        )
        self._reference_written_sample_indices: set[int] = set()
        self.train_dataset = None
        self.val_dataset = None
        self.save_hyperparameters(ignore=["model", "tokenizer", "val_samples"])

    def _sample_prefix(self, epoch: int, sample_idx: int) -> str:
        if self.cfg.sample_every_n_items is None:
            return f"sample_{epoch}_{sample_idx}"
        return f"sample_{epoch}_{int(self.global_step)}_{sample_idx}"

    def _artifact_path(self, epoch: int, sample_idx: int, label: str, suffix: str) -> Path:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        return self.cfg.output_dir / f"{self._sample_prefix(epoch, sample_idx)}.{label}.{suffix}"

    def _build_prompt(self, text: str) -> str:
        if self.cfg.language == "njam-v4":
            keep = max(8, int(len(text) * self.cfg.sample_prompt_ratio))
            return text[:keep]
        tokens = text.split()
        return " ".join(tokens[: max(8, int(len(tokens) * self.cfg.sample_prompt_ratio))])

    def _truncate_prompt_to_context_budget(self, prompt: str, reserved_new_tokens: int) -> str:
        max_positions = int(self.model.config.max_position_embeddings)
        max_prompt_tokens = max(1, max_positions - reserved_new_tokens - 1)
        if len(self.tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
            return prompt
        if self.cfg.language == "njam-v4":
            for start_idx in range(len(prompt)):
                candidate = prompt[start_idx:].strip()
                if candidate and len(self.tokenizer.encode(candidate, add_special_tokens=False)) <= max_prompt_tokens:
                    return candidate
            return prompt[-1:] if prompt else prompt
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
                logits_processor=build_generation_logits_processor(
                    self.tokenizer,
                    self.language.name,
                    enabled=self.cfg.grammar_constrained_generation,
                ),
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

    def _run_musical_eval(self) -> None:
        if not self.cfg.musical_eval:
            return
        assert self.cfg.musical_eval_every_n_epochs >= 1, "musical_eval_every_n_epochs must be at least 1."
        if self.current_epoch % self.cfg.musical_eval_every_n_epochs != 0:
            return
        result = run_musical_eval(
            model=self.model,
            tokenizer=self.tokenizer,
            language=self.language,
            max_new_tokens=self.cfg.musical_eval_max_new_tokens,
            device=self.device,
            grammar_constrained_generation=self.cfg.grammar_constrained_generation,
        )
        logger = self.logger.experiment if self.logger else None
        log_musical_eval_to_tensorboard(result, logger, int(self.current_epoch))
        write_musical_eval_json(
            result,
            self.cfg.output_dir / f"musical_eval_epoch_{int(self.current_epoch):04d}.json",
        )

    def _remove_render_bundle(self, paths: Dict[str, str]) -> None:
        for path_str in paths.values():
            path = Path(path_str)
            if path.exists():
                path.unlink()

    def _slice_model_only_document(self, generated_doc: NJamDocument, prompt: str) -> Optional[NJamDocument]:
        prompt_doc = self.language.parse_document(prompt)
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
        metadata = self.language.extract_header_metadata(prompt)
        return self.language.recover_continuation_document(model_only_text, metadata=metadata)

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
            self.language.encode_document(document),
        )
        summary["generated_model_only_parse_ok"] = True
        summary["generated_model_only_paths"] = model_only_paths
        summary["generated_model_only_render_mode"] = render_mode
        if "wav" in model_only_paths:
            self._log_sample_audio(Path(model_only_paths["wav"]), sample_idx)

    def _write_reference_once(self, sample_idx: int, text: str, summary: Dict[str, object]) -> None:
        if sample_idx in self._reference_written_sample_indices:
            summary["reference_paths"] = "written_at_epoch_0_only"
            return
        try:
            reference_doc = self.language.parse_document(text)
            summary["reference_paths"] = self._write_render_bundle(
                reference_doc,
                self.current_epoch,
                sample_idx,
                "reference",
                text,
            )
            self._reference_written_sample_indices.add(sample_idx)
        except Exception as exc:
            summary["reference_error"] = str(exc)

    def _render_validation_sample(self, sample_idx: int, sample: Dict[str, object]) -> None:
        text = str(sample["text"])
        body_text = njam_body_text(text, language=self.cfg.language)
        prompt = self._build_prompt(body_text)
        effective_prompt, full_body_text, model_only_text = self._generate_sample_text(prompt)
        self._log_sample_text(sample_idx, full_body_text, model_only_text)
        # evaluate the quality of the text we got out of the model
        model_only_recovery_stats = self.language.analyze_parseable_continuation(model_only_text).to_dict()
        self._log_sample_metrics(sample_idx, model_only_recovery_stats)
        header_text = njam_header_text(text, language=self.cfg.language)
        generated_text = header_text + "\n" + full_body_text.strip() + "\n"
        summary: Dict[str, object] = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "sample_idx": int(sample_idx),
            "prompt": effective_prompt,
            "generated_text_preview": full_body_text[:500],
            "generated_model_only_preview": model_only_text[:500],
            "generated_model_only_recovery_stats": model_only_recovery_stats,
            "generated_parse_ok": False,
            "generated_model_only_parse_ok": False,
        }
        try:
            generated_doc = self.language.parse_document(generated_text)
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

    def on_train_batch_end(self, outputs, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.cfg.sample_every_n_items is None:
            return
        if not self.val_samples or self.cfg.sample_limit <= 0:
            return
        assert self.cfg.sample_every_n_items > 0, "sample_every_n_items must be positive."
        batch_items = int(batch["input_ids"].shape[0])
        self._items_seen_for_sample_render += batch_items
        assert self._next_sample_render_item_target is not None
        if self._items_seen_for_sample_render < self._next_sample_render_item_target:
            return
        for idx, sample in enumerate(self.val_samples[: self.cfg.sample_limit]):
            self._render_validation_sample(idx, sample)
        self._next_sample_render_item_target += int(self.cfg.sample_every_n_items)

    def on_train_epoch_end(self) -> None:
        if hasattr(self.train_dataset, "advance_epoch"):
            self.train_dataset.advance_epoch()

    def on_validation_epoch_start(self) -> None:
        if hasattr(self.val_dataset, "prepare_validation_epoch"):
            self.val_dataset.prepare_validation_epoch()

    def _finish_validation_dataset_epoch(self) -> None:
        if hasattr(self.val_dataset, "finish_validation_epoch"):
            self.val_dataset.finish_validation_epoch()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        perplexity = torch.exp(loss.detach())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        try:
            self._run_musical_eval()
            if self.val_samples and self.cfg.sample_every_n_items is None:
                assert self.cfg.sample_every_n_epochs >= 1, "sample_every_n_epochs must be at least 1."
                if self.current_epoch % self.cfg.sample_every_n_epochs == 0:
                    for idx, sample in enumerate(self.val_samples[: self.cfg.sample_limit]):
                        self._render_validation_sample(idx, sample)
        finally:
            self._finish_validation_dataset_epoch()

    def configure_optimizers(self):
        assert self.cfg.lr_scheduler in {"constant", "linear", "cosine"}, "Unsupported lr_scheduler."
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=max(0.0, float(self.cfg.weight_decay)),
        )
        if self.cfg.lr_scheduler == "constant" and self.cfg.warmup_steps <= 0:
            return optimizer

        warmup_steps = max(0, int(self.cfg.warmup_steps))
        total_steps = max(1, int(getattr(self.trainer, "estimated_stepping_batches", 1) or 1))

        def lr_lambda(step: int) -> float:
            step = int(step)
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, step / max(1, warmup_steps))
            if self.cfg.lr_scheduler == "constant":
                return 1.0
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            if self.cfg.lr_scheduler == "linear":
                return max(0.0, 1.0 - progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
                "interval": "step",
                "frequency": 1,
            },
        }


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


def configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The '.*_dataloader' does not have many workers.*",
        module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
    )


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


def partial_dataloader_kwargs() -> Dict[str, object]:
    return {"num_workers": 0, "pin_memory": torch.cuda.is_available()}


def build_sliding_window_dataset(
    texts: Sequence[str],
    tokenizer: SentencePieceTokenizerAdapter,
    seq_len: int,
    batch_size: int,
    split_name: str,
    mode: str,
    prep_workers: Optional[int],
    language: str,
    musical_window_bars: int = 4,
    musical_window_hop_bars: int = 1,
    min_window_notes: int = 8,
):
    """Construct the requested train/validation dataset implementation.

    Args:
        texts: Full NJam documents for one split.
        tokenizer: SentencePiece tokenizer adapter for the run.
        seq_len: Model context length used for samples.
        batch_size: DataLoader batch size; partial modes expose one batch per solo.
        split_name: Human-readable split label for progress/errors.
        mode: ``full``, ``partial``, ``partial-random``, or musical variants.
        prep_workers: Worker count for exhaustive token-window preparation.
        language: Active NJam language name.
    """
    if mode == "full":
        return SoloSlidingWindowDataset(
            texts,
            tokenizer,
            seq_len,
            split_name=split_name,
            prep_workers=prep_workers,
            language=language,
        )
    if mode in {"partial", "partial-random"}:
        return SoloSlidingWindowDatasetPartial(
            texts,
            tokenizer,
            seq_len,
            batch_size=batch_size,
            split_name=split_name,
            prep_workers=prep_workers,
            randomize_each_epoch=mode == "partial-random",
            language=language,
        )
    if mode in {"musical-partial", "musical-partial-random"}:
        return SoloMusicalWindowDatasetPartial(
            texts,
            tokenizer,
            seq_len,
            batch_size=batch_size,
            split_name=split_name,
            randomize_each_epoch=mode == "musical-partial-random",
            language=language,
            musical_window_bars=musical_window_bars,
            musical_window_hop_bars=musical_window_hop_bars,
            min_window_notes=min_window_notes,
        )
    raise AssertionError(f"Unsupported sliding-window dataset mode: {mode!r}")


def dataset_dataloader_kwargs(dataset) -> Dict[str, object]:
    if isinstance(dataset, (SoloSlidingWindowDatasetPartial, SoloMusicalWindowDatasetPartial)):
        return partial_dataloader_kwargs()
    return dataloader_kwargs()


def run_training(config: TrainConfig) -> Dict[str, object]:
    configure_warning_filters()
    configure_torch_runtime()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    language = get_language(config.language)
    assert config.dataset_mode in {
        "partial",
        "full",
        "musical-partial",
    }, "dataset_mode must be one of: partial, full, musical-partial."
    assert config.validation_dataset_mode in {
        "partial-random",
        "partial",
        "full",
        "musical-partial",
        "musical-partial-random",
    }, "validation_dataset_mode must be one of: partial-random, partial, full, musical-partial, musical-partial-random."
    assert config.gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be at least 1."
    assert config.max_grad_norm >= 0, "max_grad_norm must be non-negative."
    assert config.weight_decay >= 0, "weight_decay must be non-negative."
    assert config.warmup_steps >= 0, "warmup_steps must be non-negative."
    assert config.lr_scheduler in {"constant", "linear", "cosine"}, "lr_scheduler must be one of: constant, linear, cosine."
    records = load_corpus_records(config.corpus_path)
    splits = split_records_by_solo(records)
    tokenizer_dir = config.output_dir / "tokenizer"
    tokenizer_model_type = getattr(language, "tokenizer_model_type", lambda: "unigram")()
    tokenizer_train_kwargs = getattr(language, "tokenizer_train_kwargs", lambda: {})()
    tokenizer_texts = [language.body_text(str(record["text"])) for record in splits["train"]]
    tokenizer_texts.extend(getattr(language, "tokenizer_seed_texts", lambda: [])())
    print(f"LOG about to build the tokenizer")
    tokenizer = build_sentencepiece_tokenizer(
        tokenizer_texts,
        tokenizer_dir,
        model_type=tokenizer_model_type,
        trainer_kwargs=tokenizer_train_kwargs,
    )
    tokenizer.loss_mask_token_ids = set(getattr(language, "loss_mask_token_ids", lambda _tokenizer: set())(tokenizer))
    tokenizer.tokenizer_model_type = tokenizer_model_type
    tokenizer.save_pretrained(str(tokenizer_dir))
    train_ds = build_sliding_window_dataset(
        [str(r["text"]) for r in splits["train"]],
        tokenizer,
        config.seq_len,
        batch_size=config.batch_size,
        split_name="train",
        mode=config.dataset_mode,
        prep_workers=config.dataset_prep_workers,
        language=language.name,
        musical_window_bars=config.musical_window_bars,
        musical_window_hop_bars=config.musical_window_hop_bars,
        min_window_notes=config.min_window_notes,
    )
    val_ds = build_sliding_window_dataset(
        [str(r["text"]) for r in splits["val"]],
        tokenizer,
        config.seq_len,
        batch_size=config.batch_size,
        split_name="val",
        mode=config.validation_dataset_mode,
        prep_workers=config.dataset_prep_workers,
        language=language.name,
        musical_window_bars=config.musical_window_bars,
        musical_window_hop_bars=config.musical_window_hop_bars,
        min_window_notes=config.min_window_notes,
    )
    test_ds = build_sliding_window_dataset(
        [str(r["text"]) for r in splits["test"]],
        tokenizer,
        config.seq_len,
        batch_size=config.batch_size,
        split_name="test",
        mode=config.validation_dataset_mode,
        prep_workers=config.dataset_prep_workers,
        language=language.name,
        musical_window_bars=config.musical_window_bars,
        musical_window_hop_bars=config.musical_window_hop_bars,
        min_window_notes=config.min_window_notes,
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
    module.train_dataset = train_ds
    module.val_dataset = val_ds
    logger = TensorBoardLogger(save_dir=str(config.output_dir), name="tensorboard")
    checkpoint = ModelCheckpoint(
        dirpath=str(config.output_dir / "checkpoints"),
        filename="best",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_on_train_epoch_end=False,
    )
    assert config.early_stopping_patience >= 0, "early_stopping_patience must be non-negative."
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.early_stopping_patience,
    )
    def make_train_loader() -> DataLoader:
        return DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=not isinstance(train_ds, (SoloSlidingWindowDatasetPartial, SoloMusicalWindowDatasetPartial)),
            **dataset_dataloader_kwargs(train_ds),
        )

    def make_val_loader() -> DataLoader:
        return DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            **dataset_dataloader_kwargs(val_ds),
        )

    accelerator_kwargs = detect_accelerator()
    if config.validation_preflight:
        assert config.validation_preflight_val_batches >= 1, "validation_preflight_val_batches must be at least 1."
        preflight_trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=config.validation_preflight_val_batches,
            num_sanity_val_steps=0,
            logger=logger,
            callbacks=[],
            enable_checkpointing=False,
            log_every_n_steps=1,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            gradient_clip_val=float(config.max_grad_norm),
            **accelerator_kwargs,
        )
        preflight_trainer.fit(module, make_train_loader(), make_val_loader())
        for dataset in (train_ds, val_ds):
            if hasattr(dataset, "reset_cursors"):
                dataset.reset_cursors()

    trainer_kwargs = {"max_epochs": config.max_epochs}
    if config.validation_preflight:
        trainer_kwargs["num_sanity_val_steps"] = 0

    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint, early_stop],
        enable_checkpointing=True,
        log_every_n_steps=1,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=float(config.max_grad_norm),
        **trainer_kwargs,
        **accelerator_kwargs,
    )
    trainer.fit(module, make_train_loader(), make_val_loader())
    hf_dir = config.output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))
    summary = {
        "best_model_path": checkpoint.best_model_path,
        "hf_model_dir": str(hf_dir),
        "language": language.name,
        "dataset_mode": config.dataset_mode,
        "validation_dataset_mode": config.validation_dataset_mode,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "train_possible_windows": sum(train_ds.window_counts_per_solo),
        "val_possible_windows": sum(val_ds.window_counts_per_solo),
        "test_possible_windows": sum(test_ds.window_counts_per_solo),
        "train_epoch_steps": math.ceil(len(train_ds) / config.batch_size),
        "val_epoch_steps": math.ceil(len(val_ds) / config.batch_size),
        "mean_train_windows_per_solo": (sum(train_ds.window_counts_per_solo) / len(train_ds.window_counts_per_solo)),
        "mean_val_windows_per_solo": (sum(val_ds.window_counts_per_solo) / len(val_ds.window_counts_per_solo)),
        "mean_test_windows_per_solo": (sum(test_ds.window_counts_per_solo) / len(test_ds.window_counts_per_solo)),
        "train_dataset_build_stats": getattr(train_ds, "build_stats", None),
        "val_dataset_build_stats": getattr(val_ds, "build_stats", None),
        "test_dataset_build_stats": getattr(test_ds, "build_stats", None),
        "window_stride": 1,
        "header_tokens_dropped": True,
        "left_padding": True,
        "pad_loss_masked": True,
        "sample_every_n_items": config.sample_every_n_items,
        "validation_preflight": config.validation_preflight,
        "validation_preflight_val_batches": config.validation_preflight_val_batches,
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": config.early_stopping_patience,
        "sample_step_interval_estimate": (
            None if config.sample_every_n_items is None else math.ceil(config.sample_every_n_items / config.batch_size)
        ),
        "dataset_prep_workers": config.dataset_prep_workers,
        "tokenizer_dir": str(config.output_dir / "tokenizer"),
        "tokenizer_model_type": tokenizer_model_type,
        "loss_mask_token_ids": sorted(int(token_id) for token_id in tokenizer.loss_mask_token_ids),
        "musical_eval": config.musical_eval,
        "musical_eval_every_n_epochs": config.musical_eval_every_n_epochs,
        "musical_eval_max_new_tokens": config.musical_eval_max_new_tokens,
        "grammar_constrained_generation": config.grammar_constrained_generation,
        "musical_window_bars": config.musical_window_bars,
        "musical_window_hop_bars": config.musical_window_hop_bars,
        "min_window_notes": config.min_window_notes,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "warmup_steps": config.warmup_steps,
        "lr_scheduler": config.lr_scheduler,
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
