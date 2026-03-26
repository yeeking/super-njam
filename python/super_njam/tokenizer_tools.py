"""Tokenizer experiments for NJamV3 corpora."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from tokenizers.trainers import BpeTrainer, WordLevelTrainer


@dataclass
class TokenizerReport:
    name: str
    total_tokens: int
    mean_tokens_per_sample: float
    mean_tokens_per_event: float
    vocab_size: int
    deterministic_roundtrip: bool
    preview: List[Dict[str, object]]


def load_corpus_texts(corpus_path: Path) -> List[str]:
    assert corpus_path.exists(), f"Corpus file does not exist: {corpus_path}"
    texts = []
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        texts.append(obj["text"])
    assert texts, f"No texts found in corpus file: {corpus_path}"
    return texts


def _event_count(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0
    return sum(1 for token in " ".join(lines[1:]).split() if token and token[0] != "T")


def _preview_items(texts: Sequence[str], encode: Callable[[str], List[str]], limit: int = 3) -> List[Dict[str, object]]:
    items = []
    for text in texts[:limit]:
        items.append({"text": text[:160], "tokens": encode(text)[:32]})
    return items


def compare_tokenizers(corpus_path: Path, output_path: Path, vocab_size: int = 512) -> List[TokenizerReport]:
    texts = load_corpus_texts(corpus_path)
    event_counts = [_event_count(text) for text in texts]

    def character_encode(text: str) -> List[str]:
        return list(text)

    def whitespace_encode(text: str) -> List[str]:
        return text.split()

    bpe = Tokenizer(BPE(unk_token="[UNK]"))
    bpe.pre_tokenizer = Whitespace()
    bpe_trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[BOS]", "[EOS]"])
    bpe.train_from_iterator(texts, trainer=bpe_trainer)

    wordlevel = Tokenizer(WordLevel(unk_token="[UNK]"))
    wordlevel.pre_tokenizer = WhitespaceSplit()
    wordlevel_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]"])
    wordlevel.train_from_iterator(texts, trainer=wordlevel_trainer)

    spm_prefix = output_path.with_suffix("")
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=str(spm_prefix),
        vocab_size=max(128, vocab_size),
        model_type="unigram",
        max_sentence_length=65536,
        hard_vocab_limit=False,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        train_extremely_large_corpus=False,
    )
    sp = spm.SentencePieceProcessor(model_file=str(spm_prefix) + ".model")

    configs = [
        (
            "character",
            character_encode,
            lambda text: "".join(character_encode(text)) == text,
            256,
        ),
        (
            "whitespace",
            whitespace_encode,
            lambda text: " ".join(whitespace_encode(text)) == " ".join(text.split()),
            0,
        ),
        (
            "bpe",
            lambda text: bpe.encode(text).tokens,
            lambda text: bpe.decode(bpe.encode(text).ids) == text,
            bpe.get_vocab_size(),
        ),
        (
            "custom_wordlevel",
            lambda text: wordlevel.encode(text).tokens,
            lambda text: wordlevel.decode(wordlevel.encode(text).ids) == text,
            wordlevel.get_vocab_size(),
        ),
        (
            "sentencepiece_unigram",
            lambda text: sp.encode(text, out_type=str),
            lambda text: sp.decode(sp.encode(text, out_type=int)) == text,
            int(sp.get_piece_size()),
        ),
    ]

    reports = []
    for name, encoder, roundtrip_fn, vocab in configs:
        token_counts = [len(encoder(text)) for text in texts]
        deterministic = all(roundtrip_fn(text) for text in texts[: min(5, len(texts))])
        reports.append(
            TokenizerReport(
                name=name,
                total_tokens=sum(token_counts),
                mean_tokens_per_sample=statistics.mean(token_counts),
                mean_tokens_per_event=statistics.mean(
                    count / max(events, 1) for count, events in zip(token_counts, event_counts)
                ),
                vocab_size=vocab,
                deterministic_roundtrip=deterministic,
                preview=_preview_items(texts, encoder),
            )
        )
    output_path.write_text(json.dumps([asdict(report) for report in reports], indent=2) + "\n")
    return reports
