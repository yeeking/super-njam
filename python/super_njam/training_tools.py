"""Training utilities for llama-compatible NJam models."""

from __future__ import annotations

import hashlib
import json
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

from .audio_tools import render_document_audio
from .midi_tools import write_midi
from .njam_v3 import NJamDocument, encode_document, parse_document

DEFAULT_TRAINING_SOUNDFONT = Path("soundfonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2")


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
        max_sentence_length=65536,
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


class PackedCausalDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: SentencePieceTokenizerAdapter, seq_len: int):
        assert texts, "PackedCausalDataset requires non-empty texts."
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        assert bos is not None and eos is not None, "Tokenizer must define BOS and EOS tokens."
        token_ids: List[int] = []
        for text in texts:
            token_ids.extend([bos] + tokenizer.encode(text, add_special_tokens=False) + [eos])
        assert len(token_ids) > seq_len, "Not enough tokens for the requested sequence length."
        self.chunks = []
        for start in range(0, len(token_ids) - seq_len, seq_len):
            chunk = token_ids[start : start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
        assert self.chunks, "Token packing produced zero chunks."

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


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

    def _generate_sample_text(self, prompt: str) -> Tuple[str, str]:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        max_positions = int(self.model.config.max_position_embeddings)
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
        return full_text, continuation_text

    def _write_sample_midi(self, document, midi_path: Path) -> None:
        render_doc = NJamDocument(
            metadata={**document.metadata, "render_instrument": self.cfg.render_instrument},
            events=list(document.events),
        )
        write_midi(render_doc, midi_path)

    def _write_sample_audio(self, document, wav_path: Path) -> None:
        render_doc = NJamDocument(
            metadata={**document.metadata, "render_instrument": self.cfg.render_instrument},
            events=list(document.events),
        )
        render_document_audio(render_doc, wav_path, soundfont_path=self.cfg.soundfont_path)

    def _write_render_bundle(self, document, epoch: int, sample_idx: int, label: str, text_out: str) -> Dict[str, str]:
        njam_path = self._artifact_path(epoch, sample_idx, label, "njam")
        midi_path = self._artifact_path(epoch, sample_idx, label, "mid")
        wav_path = self._artifact_path(epoch, sample_idx, label, "wav")
        njam_path.write_text(text_out, encoding="utf-8")
        self._write_sample_midi(document, midi_path)
        self._write_sample_audio(document, wav_path)
        return {
            "njam": str(njam_path),
            "midi": str(midi_path),
            "wav": str(wav_path),
        }

    def _log_sample_audio(self, wav_path: Path, sample_idx: int) -> None:
        logger = self.logger.experiment if self.logger else None
        if logger is None or not wav_path.exists():
            return
        audio_tensor = torch.tensor(_read_wav_mono(wav_path), dtype=torch.float32).unsqueeze(0)
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

    def _write_sample_summary(self, sample_idx: int, payload: Dict[str, object]) -> None:
        summary_path = self._artifact_path(self.current_epoch, sample_idx, "summary", "json")
        summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

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
        prompt = self._build_prompt(text)
        full_text, model_only_text = self._generate_sample_text(prompt)
        self._log_sample_text(sample_idx, full_text, model_only_text)
        summary: Dict[str, object] = {
            "epoch": int(self.current_epoch),
            "sample_idx": int(sample_idx),
            "prompt": prompt,
            "generated_text_preview": full_text[:500],
            "generated_model_only_preview": model_only_text[:500],
            "generated_parse_ok": False,
            "generated_model_only_parse_ok": False,
        }
        try:
            generated_doc = parse_document(full_text)
            generated_paths = self._write_render_bundle(generated_doc, self.current_epoch, sample_idx, "generated_full", full_text)
            summary["generated_parse_ok"] = True
            summary["generated_full_paths"] = generated_paths
            self._log_sample_audio(Path(generated_paths["wav"]), sample_idx)
            model_only_doc = self._slice_model_only_document(generated_doc, prompt)
            if model_only_doc is not None and model_only_doc.events:
                model_only_paths = self._write_render_bundle(
                    model_only_doc,
                    self.current_epoch,
                    sample_idx,
                    "generated_model_only",
                    encode_document(model_only_doc),
                )
                summary["generated_model_only_parse_ok"] = True
                summary["generated_model_only_paths"] = model_only_paths
        except Exception as exc:
            raw_generated_path = self._artifact_path(self.current_epoch, sample_idx, "generated_raw", "txt")
            raw_generated_path.write_text(full_text, encoding="utf-8")
            raw_model_only_path = self._artifact_path(self.current_epoch, sample_idx, "generated_model_only_raw", "txt")
            raw_model_only_path.write_text(model_only_text, encoding="utf-8")
            summary["generated_error"] = str(exc)
            summary["generated_raw_path"] = str(raw_generated_path)
            summary["generated_model_only_raw_path"] = str(raw_model_only_path)
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


def detect_accelerator() -> Dict[str, object]:
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


def run_training(config: TrainConfig) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_corpus_records(config.corpus_path)
    splits = split_records_by_solo(records)
    tokenizer_dir = config.output_dir / "tokenizer"
    tokenizer = build_sentencepiece_tokenizer([record["text"] for record in splits["train"]], tokenizer_dir)
    tokenizer.save_pretrained(str(tokenizer_dir))
    train_ds = PackedCausalDataset([r["text"] for r in splits["train"]], tokenizer, config.seq_len)
    val_ds = PackedCausalDataset([r["text"] for r in splits["val"]], tokenizer, config.seq_len)

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
        save_top_k=1,
        monitor="val_loss",
        mode="min",
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
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=config.batch_size),
    )
    hf_dir = config.output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))
    summary = {
        "best_model_path": checkpoint.best_model_path,
        "hf_model_dir": str(hf_dir),
        "train_chunks": len(train_ds),
        "val_chunks": len(val_ds),
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
