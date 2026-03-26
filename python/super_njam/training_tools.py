"""Training utilities for llama-compatible NJam models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import lightning as L
import sentencepiece as spm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerBase

from .audio_tools import render_document_audio
from .midi_tools import write_midi
from .njam_v3 import parse_document


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
) -> LlamaTokenizer:
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
    tokenizer = LlamaTokenizer(vocab_file=str(model_prefix) + ".model")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class PackedCausalDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: PreTrainedTokenizerBase, seq_len: int):
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
    soundfont_path: Optional[Path] = None


class NJamLightningModule(L.LightningModule):
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        val_samples: Sequence[Dict[str, object]],
        config: TrainConfig,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.val_samples = list(val_samples)
        self.cfg = config
        self.save_hyperparameters(ignore=["model", "tokenizer", "val_samples"])

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
        logger = self.logger.experiment if self.logger else None
        for idx, sample in enumerate(self.val_samples[: self.cfg.sample_limit]):
            text = str(sample["text"])
            prompt = " ".join(text.split()[: max(8, int(len(text.split()) * self.cfg.sample_prompt_ratio))])
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
            text_out = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            if logger is not None:
                logger.add_text(f"samples/sample_{idx}", text_out, self.current_epoch)
            try:
                doc = parse_document(text_out)
                midi_path = self.cfg.output_dir / f"sample_{self.current_epoch}_{idx}.mid"
                wav_path = self.cfg.output_dir / f"sample_{self.current_epoch}_{idx}.wav"
                write_midi(doc, midi_path)
                render_document_audio(doc, wav_path, soundfont_path=self.cfg.soundfont_path)
                if logger is not None and wav_path.exists():
                    audio_tensor = torch.tensor(_read_wav_mono(wav_path), dtype=torch.float32).unsqueeze(0)
                    logger.add_audio(f"samples_audio/sample_{idx}", audio_tensor, self.current_epoch, sample_rate=22050)
            except Exception as exc:
                if logger is not None:
                    logger.add_text(f"samples/sample_{idx}_error", str(exc), self.current_epoch)

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
