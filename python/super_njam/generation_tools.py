"""Generation helpers for NJam models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence

import torch
from transformers import LogitsProcessor

from . import njam_v4


FREE = "free"
NOTE_PITCH = "note_pitch"
NOTE_VELOCITY = "note_velocity"
NOTE_DURATION = "note_duration"
NOTE_DURATION_TAIL = "note_duration_tail"
CC_NUMBER = "cc_number"
CC_VALUE = "cc_value"
BEND_VALUE = "bend_value"
PROGRAM_VALUE = "program_value"
GRAMMAR_STATES = (
    FREE,
    NOTE_PITCH,
    NOTE_VELOCITY,
    NOTE_DURATION,
    NOTE_DURATION_TAIL,
    CC_NUMBER,
    CC_VALUE,
    BEND_VALUE,
    PROGRAM_VALUE,
)


def _base_token_ids(text: str) -> List[int]:
    token_ids = [njam_v4._token_id(ch) for ch in text]
    return [int(token_id) for token_id in token_ids if token_id is not None]


def _is_time(token_id: int) -> bool:
    return njam_v4._chunk_value(token_id, njam_v4.TIME_BASE) is not None


def _is_duration(token_id: int) -> bool:
    return njam_v4._chunk_value(token_id, njam_v4.DURATION_BASE) is not None


def advance_njam_v4_grammar_state(state: str, token_id: int) -> str | None:
    """Advance the NJam-v4 grammar state by one base token.

    Returns ``None`` when the token is illegal in the current state.
    """

    if state == NOTE_DURATION_TAIL and not _is_duration(token_id):
        state = FREE

    if state == NOTE_PITCH:
        return NOTE_VELOCITY if njam_v4._is_pitch(token_id) else None
    if state == NOTE_VELOCITY:
        return NOTE_DURATION if njam_v4._is_velocity(token_id) else None
    if state == NOTE_DURATION:
        return NOTE_DURATION_TAIL if _is_duration(token_id) else None
    if state == NOTE_DURATION_TAIL:
        return NOTE_DURATION_TAIL if _is_duration(token_id) else None
    if state == CC_NUMBER:
        return CC_VALUE if njam_v4._is_cc_number(token_id) else None
    if state == CC_VALUE:
        return FREE if njam_v4._is_cc_value(token_id) else None
    if state == BEND_VALUE:
        return FREE if njam_v4._is_bend(token_id) else None
    if state == PROGRAM_VALUE:
        return FREE if njam_v4._is_program(token_id) else None

    if state != FREE:
        return None

    if token_id == njam_v4.NOTE:
        return NOTE_PITCH
    if token_id == njam_v4.CC:
        return CC_NUMBER
    if token_id == njam_v4.BEND:
        return BEND_VALUE
    if token_id == njam_v4.PROGRAM:
        return PROGRAM_VALUE
    if token_id in {njam_v4.START, njam_v4.END}:
        return FREE
    if njam_v4._is_control(token_id) or _is_time(token_id):
        return FREE
    return None


def njam_v4_grammar_state_for_text(text: str) -> str:
    state = FREE
    for token_id in _base_token_ids(text):
        next_state = advance_njam_v4_grammar_state(state, token_id)
        if next_state is None:
            state = FREE
        else:
            state = next_state
    return state


def njam_v4_piece_is_allowed(base_token_ids: Sequence[int], state: str) -> bool:
    """Return whether a full SentencePiece piece is legal from ``state``.

    Args:
        base_token_ids: NJam-v4 base-token ids contained inside one tokenizer piece.
        state: Grammar state before the piece is emitted.
    """
    if not base_token_ids:
        return state == FREE
    current = state
    for token_id in base_token_ids:
        next_state = advance_njam_v4_grammar_state(current, int(token_id))
        if next_state is None:
            return False
        current = next_state
    return True


@dataclass(frozen=True)
class NJamV4PieceInfo:
    token_id: int
    base_token_ids: tuple[int, ...]


def _piece_text(tokenizer, token_id: int) -> str:
    processor = getattr(tokenizer, "processor", None)
    if processor is not None:
        try:
            return str(processor.id_to_piece(int(token_id)))
        except Exception:
            pass
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return ""


@lru_cache(maxsize=32)
def _piece_infos_for_tokenizer(model_path: str, vocab_size: int) -> tuple[NJamV4PieceInfo, ...]:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=model_path)
    infos = []
    for token_id in range(vocab_size):
        piece = str(processor.id_to_piece(token_id))
        infos.append(NJamV4PieceInfo(token_id=token_id, base_token_ids=tuple(_base_token_ids(piece))))
    return tuple(infos)


def njam_v4_piece_infos(tokenizer) -> tuple[NJamV4PieceInfo, ...]:
    model_path = getattr(tokenizer, "model_path", None)
    vocab_size = int(getattr(tokenizer, "vocab_size"))
    if model_path is not None:
        return _piece_infos_for_tokenizer(str(model_path), vocab_size)
    return tuple(
        NJamV4PieceInfo(token_id=token_id, base_token_ids=tuple(_base_token_ids(_piece_text(tokenizer, token_id))))
        for token_id in range(vocab_size)
    )


class NJamV4GrammarLogitsProcessor(LogitsProcessor):
    """Mask SentencePiece tokens that would violate NJam-v4 base-token grammar."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.piece_infos = njam_v4_piece_infos(tokenizer)
        self.base_token_ids_by_token_id = [info.base_token_ids for info in self.piece_infos]
        self.eos_token_id = int(getattr(tokenizer, "eos_token_id", -1))
        self._allowed_mask_cache: dict[tuple[str, str, int], torch.Tensor] = {}

    def _state_for_token_ids(self, token_ids: Sequence[int]) -> str:
        state = FREE
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.base_token_ids_by_token_id):
                continue
            for base_token_id in self.base_token_ids_by_token_id[int(token_id)]:
                next_state = advance_njam_v4_grammar_state(state, base_token_id)
                if next_state is None:
                    state = FREE
                else:
                    state = next_state
        return state

    def _allowed_mask(self, state: str, device: torch.device, vocab_size: int) -> torch.Tensor:
        cache_key = (state, str(device), int(vocab_size))
        cached = self._allowed_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for info in self.piece_infos[:vocab_size]:
            if info.token_id == self.eos_token_id:
                allowed[info.token_id] = state == FREE
                continue
            if njam_v4_piece_is_allowed(info.base_token_ids, state):
                allowed[info.token_id] = True
        if not bool(allowed.any()):
            allowed[:] = True
        self._allowed_mask_cache[cache_key] = allowed
        return allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = int(scores.shape[-1])
        row_masks = []
        for row_idx in range(int(input_ids.shape[0])):
            state = self._state_for_token_ids(input_ids[row_idx].detach().cpu().tolist())
            row_masks.append(self._allowed_mask(state, scores.device, vocab_size))
        allowed = torch.stack(row_masks, dim=0)
        return scores.masked_fill(~allowed, -float("inf"))


def build_generation_logits_processor(tokenizer, language_name: str, enabled: bool = True):
    """Build optional generation constraints for the active language.

    Args:
        tokenizer: Project tokenizer adapter or compatible tokenizer.
        language_name: Language adapter name, e.g. ``njam-v4``.
        enabled: False returns no processors, preserving unconstrained generation.
    """
    if enabled and language_name == "njam-v4":
        return [NJamV4GrammarLogitsProcessor(tokenizer)]
    return None
