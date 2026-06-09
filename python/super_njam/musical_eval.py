"""Synthetic musical evaluation prompts and metrics for NJam models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .generation_tools import build_generation_logits_processor
from .music_language import MusicLanguage, get_language
from .njam_v3 import NJamDocument, NoteEvent


DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_MIN_NOTES = 4
DEFAULT_PPQ = 96


@dataclass(frozen=True)
class MusicalEvalCase:
    name: str
    category: str
    document: NJamDocument
    target_pitch_classes: Optional[set[int]] = None
    target_onsets: Optional[List[int]] = None
    target_iois: Optional[List[int]] = None


@dataclass
class MusicalEvalCaseResult:
    name: str
    category: str
    prompt_token_count: int
    generated_token_count: int
    recovered_event_count: int
    recovered_note_count: int
    parseable_note_rate: float
    note_marker_count: int
    complete_note_pattern_count: int
    complete_note_pattern_rate: float
    recovery_rate: float
    recovery_quality_score: float
    scale_adherence: float
    prompt_pitch_coverage: float
    out_of_scale_rate: float
    rhythm_ioi_similarity: float
    rhythm_alignment: float
    rhythm_bar_phase_similarity: float
    overall: float
    generated_preview: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class MusicalEvalRunResult:
    language: str
    suite: str
    max_new_tokens: int
    min_notes: int
    cases: List[MusicalEvalCaseResult]
    aggregates: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "language": self.language,
            "suite": self.suite,
            "max_new_tokens": self.max_new_tokens,
            "min_notes": self.min_notes,
            "aggregates": self.aggregates,
            "cases": [case.to_dict() for case in self.cases],
        }


def _scale_pitch_classes(root: int, intervals: Sequence[int]) -> set[int]:
    return {(root + interval) % 12 for interval in intervals}


MAJOR = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR = (0, 2, 3, 5, 7, 8, 10)
MIXOLYDIAN = (0, 2, 4, 5, 7, 9, 10)


def _note_doc(pitches: Sequence[int], onsets: Sequence[int], durations: Sequence[int] | int, ppq: int = DEFAULT_PPQ) -> NJamDocument:
    if isinstance(durations, int):
        duration_values = [durations for _ in pitches]
    else:
        duration_values = list(durations)
    events = [
        NoteEvent(time=int(onset), pitch=int(pitch), velocity=88, duration=max(1, int(duration)))
        for pitch, onset, duration in zip(pitches, onsets, duration_values)
    ]
    return NJamDocument(metadata={"ppq": str(ppq), "tempo": "120.000", "sig": "4/4"}, events=events)


def _iois(onsets: Sequence[int]) -> List[int]:
    return [max(0, int(b) - int(a)) for a, b in zip(onsets, onsets[1:])]


def light_eval_cases(ppq: int = DEFAULT_PPQ) -> List[MusicalEvalCase]:
    eighth = ppq // 2
    quarter = ppq
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    a_minor = [69, 71, 72, 74, 76, 77, 79, 81]
    g_mix = [67, 69, 71, 72, 74, 76, 77, 79]
    cases: List[MusicalEvalCase] = []

    harmony_specs = [
        ("c_major_ascending", c_major, _scale_pitch_classes(0, MAJOR)),
        ("c_major_descending", list(reversed(c_major)), _scale_pitch_classes(0, MAJOR)),
        ("c_major_randomized", [64, 60, 69, 62, 71, 65, 72, 67], _scale_pitch_classes(0, MAJOR)),
        ("a_minor_ascending", a_minor, _scale_pitch_classes(9, NATURAL_MINOR)),
        ("a_minor_descending", list(reversed(a_minor)), _scale_pitch_classes(9, NATURAL_MINOR)),
        ("g_mixolydian_randomized", [74, 67, 76, 71, 69, 79, 72, 77], _scale_pitch_classes(7, MIXOLYDIAN)),
    ]
    for name, pitches, pcs in harmony_specs:
        onsets = [idx * eighth for idx in range(len(pitches))]
        cases.append(
            MusicalEvalCase(
                name=name,
                category="harmony",
                document=_note_doc(pitches, onsets, durations=eighth),
                target_pitch_classes=pcs,
            )
        )

    rhythm_specs = [
        ("quarter_note_pulse", [0, quarter, 2 * quarter, 3 * quarter, 4 * quarter, 5 * quarter, 6 * quarter, 7 * quarter]),
        ("eighth_note_pulse", [idx * eighth for idx in range(12)]),
        ("swung_long_short_eighths", [0, 64, 96, 160, 192, 256, 288, 352, 384, 448, 480, 544]),
        ("syncopated_two_bar_pattern", [0, 72, 144, 240, 336, 384, 456, 528, 624, 720]),
    ]
    rhythm_pitches = [60, 62, 64, 67, 69, 67, 64, 62, 60, 62, 64, 67]
    for name, onsets in rhythm_specs:
        pitches = rhythm_pitches[: len(onsets)]
        durations = [max(12, min(eighth, ioi)) for ioi in (_iois(onsets) + [eighth])]
        cases.append(
            MusicalEvalCase(
                name=name,
                category="rhythm",
                document=_note_doc(pitches, onsets, durations=durations),
                target_onsets=list(onsets),
                target_iois=_iois(onsets),
            )
        )
    return cases


def get_eval_cases(suite: str = "light") -> List[MusicalEvalCase]:
    if suite != "light":
        raise AssertionError(f"Unsupported musical eval suite {suite!r}. Expected: light.")
    return light_eval_cases()


def _notes(document: Optional[NJamDocument]) -> List[NoteEvent]:
    if document is None:
        return []
    return [event for event in document.sorted_events() if isinstance(event, NoteEvent)]


def _hist(values: Sequence[int], bin_size: int) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for value in values:
        key = int(round(value / bin_size))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _hist_similarity(a: Sequence[int], b: Sequence[int], bin_size: int) -> float:
    if not a or not b:
        return 0.0
    ha = _hist(a, bin_size)
    hb = _hist(b, bin_size)
    keys = set(ha) | set(hb)
    total_a = sum(ha.values())
    total_b = sum(hb.values())
    distance = sum(abs((ha.get(key, 0) / total_a) - (hb.get(key, 0) / total_b)) for key in keys)
    return max(0.0, 1.0 - (distance / 2.0))


def _alignment_score(target_onsets: Sequence[int], generated_onsets: Sequence[int], ppq: int) -> float:
    if not target_onsets or not generated_onsets:
        return 0.0
    bar_ticks = ppq * 4
    target_phase = [onset % bar_ticks for onset in target_onsets]
    generated_phase = [onset % bar_ticks for onset in generated_onsets]
    tolerance = max(3, ppq // 12)
    best = 0.0
    for offset in range(0, bar_ticks, max(1, tolerance)):
        hits = 0
        for onset in generated_phase:
            shifted = (onset + offset) % bar_ticks
            distance = min(abs(shifted - target) for target in target_phase)
            distance = min(distance, bar_ticks - distance)
            if distance <= tolerance:
                hits += 1
        best = max(best, hits / len(generated_phase))
    return best


def _pitch_metrics(case: MusicalEvalCase, notes: Sequence[NoteEvent], enough_notes: bool) -> tuple[float, float, float]:
    if not enough_notes or not case.target_pitch_classes:
        return 0.0, 0.0, 0.0
    generated_pcs = [note.pitch % 12 for note in notes]
    in_scale = sum(1 for pc in generated_pcs if pc in case.target_pitch_classes)
    scale_adherence = in_scale / len(generated_pcs) if generated_pcs else 0.0
    prompt_pitch_coverage = len(set(generated_pcs) & case.target_pitch_classes) / len(case.target_pitch_classes)
    return scale_adherence, prompt_pitch_coverage, 1.0 - scale_adherence


def _rhythm_metrics(case: MusicalEvalCase, notes: Sequence[NoteEvent], ppq: int, enough_notes: bool) -> tuple[float, float, float]:
    if not enough_notes or not case.target_onsets:
        return 0.0, 0.0, 0.0
    onsets = [int(note.time) for note in notes]
    generated_iois = _iois(onsets)
    target_iois = case.target_iois or _iois(case.target_onsets)
    ioi_similarity = _hist_similarity(target_iois, generated_iois, bin_size=max(1, ppq // 12))
    alignment = _alignment_score(case.target_onsets, onsets, ppq=ppq)
    bar_phase_similarity = _hist_similarity(
        [onset % (ppq * 4) for onset in case.target_onsets],
        [onset % (ppq * 4) for onset in onsets],
        bin_size=max(1, ppq // 12),
    )
    return ioi_similarity, alignment, bar_phase_similarity


def note_pattern_metrics(text: str, language_name: str = "njam-v4") -> Dict[str, float | int]:
    if language_name != "njam-v4":
        return {
            "note_marker_count": 0,
            "complete_note_pattern_count": 0,
            "complete_note_pattern_rate": 0.0,
        }
    from . import njam_v4

    token_ids = [njam_v4._token_id(ch) for ch in text]
    token_ids = [token_id for token_id in token_ids if token_id is not None]
    note_markers = 0
    complete_patterns = 0
    for idx, token_id in enumerate(token_ids):
        if token_id != njam_v4.NOTE:
            continue
        note_markers += 1
        if idx + 3 >= len(token_ids):
            continue
        pitch_id = token_ids[idx + 1]
        velocity_id = token_ids[idx + 2]
        duration_id = token_ids[idx + 3]
        if (
            njam_v4._is_pitch(pitch_id)
            and njam_v4._is_velocity(velocity_id)
            and njam_v4._chunk_value(duration_id, njam_v4.DURATION_BASE) is not None
        ):
            complete_patterns += 1
    return {
        "note_marker_count": note_markers,
        "complete_note_pattern_count": complete_patterns,
        "complete_note_pattern_rate": complete_patterns / note_markers if note_markers else 0.0,
    }


def _truncate_prompt_ids(tokenizer, ids: List[int], max_positions: int, max_new_tokens: int) -> List[int]:
    budget = max(1, max_positions - max_new_tokens - 1)
    if len(ids) <= budget:
        return ids
    return ids[-budget:]


def prepare_eval_prompt_text(language: MusicLanguage, document: NJamDocument) -> str:
    prompt_text = language.body_text(language.encode_document(document))
    if language.name == "njam-v4":
        from . import njam_v4

        prompt_text = prompt_text.rstrip()
        end_char = njam_v4._char(njam_v4.END)
        if prompt_text.endswith(end_char):
            prompt_text = prompt_text[: -len(end_char)]
    return prompt_text


def _generate_continuation(
    model,
    tokenizer,
    language: MusicLanguage,
    prompt_text: str,
    max_new_tokens: int,
    device: torch.device,
    grammar_constrained_generation: bool = True,
) -> tuple[str, int, int]:
    max_positions = int(model.config.max_position_embeddings)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids = _truncate_prompt_ids(tokenizer, prompt_ids, max_positions=max_positions, max_new_tokens=max_new_tokens)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    available_new_tokens = max(1, min(max_new_tokens, max_positions - int(input_ids.shape[1]) - 1))
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=available_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=build_generation_logits_processor(
                tokenizer,
                language.name,
                enabled=grammar_constrained_generation,
            ),
        )
    continuation_ids = generated[0][int(input_ids.shape[1]) :]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True), int(input_ids.shape[1]), int(len(continuation_ids))


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./-]+", "_", value)


def run_musical_eval(
    model,
    tokenizer,
    language: MusicLanguage | str,
    suite: str = "light",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_notes: int = DEFAULT_MIN_NOTES,
    device: Optional[torch.device] = None,
    grammar_constrained_generation: bool = True,
) -> MusicalEvalRunResult:
    if isinstance(language, str):
        language = get_language(language)
    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    device = device or next(model.parameters()).device
    cases = get_eval_cases(suite)
    results: List[MusicalEvalCaseResult] = []
    for case in cases:
        prompt_text = prepare_eval_prompt_text(language, case.document)
        error = None
        continuation_text = ""
        prompt_token_count = 0
        generated_token_count = 0
        recovered_doc = None
        recovery_stats = {"recovery_rate": 0.0, "quality_score": 0.0, "events_recovered": 0}
        try:
            continuation_text, prompt_token_count, generated_token_count = _generate_continuation(
                model,
                tokenizer,
                language,
                prompt_text,
                max_new_tokens=max_new_tokens,
                device=device,
                grammar_constrained_generation=grammar_constrained_generation,
            )
            recovery_stats = language.analyze_parseable_continuation(continuation_text).to_dict()
            recovered_doc = language.recover_continuation_document(
                continuation_text,
                metadata=dict(case.document.metadata),
            )
        except Exception as exc:
            error = str(exc)
        notes = _notes(recovered_doc)
        note_patterns = note_pattern_metrics(continuation_text, language.name)
        enough_notes = len(notes) >= min_notes
        ppq = int(case.document.metadata.get("ppq", DEFAULT_PPQ))
        scale_adherence, prompt_pitch_coverage, out_of_scale_rate = _pitch_metrics(case, notes, enough_notes)
        rhythm_ioi_similarity, rhythm_alignment, rhythm_bar_phase_similarity = _rhythm_metrics(case, notes, ppq, enough_notes)
        category_scores = []
        if case.category == "harmony":
            category_scores.extend([scale_adherence, prompt_pitch_coverage])
        if case.category == "rhythm":
            category_scores.extend([rhythm_ioi_similarity, rhythm_alignment, rhythm_bar_phase_similarity])
        overall = sum(category_scores) / len(category_scores) if category_scores else 0.0
        recovered_events = int(recovery_stats.get("events_recovered", 0))
        parseable_note_rate = len(notes) / max(1, recovered_events)
        results.append(
            MusicalEvalCaseResult(
                name=case.name,
                category=case.category,
                prompt_token_count=prompt_token_count,
                generated_token_count=generated_token_count,
                recovered_event_count=recovered_events,
                recovered_note_count=len(notes),
                parseable_note_rate=parseable_note_rate,
                note_marker_count=int(note_patterns["note_marker_count"]),
                complete_note_pattern_count=int(note_patterns["complete_note_pattern_count"]),
                complete_note_pattern_rate=float(note_patterns["complete_note_pattern_rate"]),
                recovery_rate=float(recovery_stats.get("recovery_rate", 0.0)),
                recovery_quality_score=float(recovery_stats.get("quality_score", 0.0)),
                scale_adherence=scale_adherence,
                prompt_pitch_coverage=prompt_pitch_coverage,
                out_of_scale_rate=out_of_scale_rate,
                rhythm_ioi_similarity=rhythm_ioi_similarity,
                rhythm_alignment=rhythm_alignment,
                rhythm_bar_phase_similarity=rhythm_bar_phase_similarity,
                overall=overall,
                generated_preview=continuation_text[:240],
                error=error,
            )
        )
    if model_was_training:
        model.train()
    return MusicalEvalRunResult(
        language=language.name,
        suite=suite,
        max_new_tokens=max_new_tokens,
        min_notes=min_notes,
        cases=results,
        aggregates=aggregate_results(results),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_results(results: Sequence[MusicalEvalCaseResult]) -> Dict[str, float]:
    harmony = [result for result in results if result.category == "harmony"]
    rhythm = [result for result in results if result.category == "rhythm"]
    return {
        "overall": _mean(result.overall for result in results),
        "parseable_note_rate": _mean(result.parseable_note_rate for result in results),
        "note_marker_count_mean": _mean(float(result.note_marker_count) for result in results),
        "complete_note_pattern_count_mean": _mean(float(result.complete_note_pattern_count) for result in results),
        "complete_note_pattern_rate": _mean(result.complete_note_pattern_rate for result in results),
        "recovered_note_count_mean": _mean(float(result.recovered_note_count) for result in results),
        "recovery_rate": _mean(result.recovery_rate for result in results),
        "recovery_quality_score": _mean(result.recovery_quality_score for result in results),
        "harmony_scale_adherence": _mean(result.scale_adherence for result in harmony),
        "harmony_prompt_pitch_coverage": _mean(result.prompt_pitch_coverage for result in harmony),
        "harmony_out_of_scale_rate": _mean(result.out_of_scale_rate for result in harmony),
        "rhythm_ioi_similarity": _mean(result.rhythm_ioi_similarity for result in rhythm),
        "rhythm_alignment": _mean(result.rhythm_alignment for result in rhythm),
        "rhythm_bar_phase_similarity": _mean(result.rhythm_bar_phase_similarity for result in rhythm),
    }


def log_musical_eval_to_tensorboard(result: MusicalEvalRunResult, logger, step: int) -> None:
    if logger is None:
        return
    for key, value in result.aggregates.items():
        logger.add_scalar(f"musical_eval/{key}", value, step)
    for case in result.cases:
        prefix = f"musical_eval_cases/{_sanitize_tag(case.name)}"
        logger.add_scalar(f"{prefix}/overall", case.overall, step)
        logger.add_scalar(f"{prefix}/recovered_note_count", case.recovered_note_count, step)
        logger.add_scalar(f"{prefix}/complete_note_pattern_rate", case.complete_note_pattern_rate, step)


def write_musical_eval_json(result: MusicalEvalRunResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")


def format_musical_eval_table(result: MusicalEvalRunResult) -> str:
    lines = [
        f"Musical eval: language={result.language} suite={result.suite} max_new_tokens={result.max_new_tokens}",
        "",
        "case                         cat       notes  note%    overall  harmony  rhythm   recover",
        "---------------------------  --------  -----  -------  -------  -------  -------  -------",
    ]
    for case in result.cases:
        harmony_score = (case.scale_adherence + case.prompt_pitch_coverage) / 2.0 if case.category == "harmony" else 0.0
        rhythm_score = (
            case.rhythm_ioi_similarity + case.rhythm_alignment + case.rhythm_bar_phase_similarity
        ) / 3.0 if case.category == "rhythm" else 0.0
        lines.append(
            f"{case.name[:27]:27}  {case.category[:8]:8}  {case.recovered_note_count:5d}  "
            f"{case.complete_note_pattern_rate:7.3f}  {case.overall:7.3f}  "
            f"{harmony_score:7.3f}  {rhythm_score:7.3f}  {case.recovery_rate:7.3f}"
        )
    lines.extend(
        [
            "",
            "Aggregates:",
            *(f"  {key}: {value:.4f}" for key, value in sorted(result.aggregates.items())),
        ]
    )
    return "\n".join(lines)
