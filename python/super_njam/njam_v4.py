"""NJamV4 structured symbolic music base-token format.

NJamV4 stores music as a stream of fixed Private Use Area characters. Each
character is a musical base token, and the trainer learns SentencePiece BPE over
that stream. The source code names the tokens; the corpus body is intentionally
compact and not meant to be hand-written.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

from .njam_v3 import (
    ContinuationRecoveryStats,
    ControlChangeEvent,
    NJamDocument,
    NJamEvent,
    NoteEvent,
    PitchBendEvent,
    ProgramChangeEvent,
)

DEFAULT_PPQ = 96
HEADER_PREFIX = "NV4|"
PUA_BASE = 0xE000

START = 0
END = 1
NOTE = 2
CC = 3
BEND = 4
PROGRAM = 5

PITCH_BASE = 100
VELOCITY_BASE = 300
CC_NUMBER_BASE = 400
CC_VALUE_BASE = 600
BEND_BASE = 700
PROGRAM_BASE = 800
TIME_BASE = 1000
DURATION_BASE = 1100
ROOT_BASE = 1200
CHORD_BASE = 1220

VELOCITY_BINS = 24
CC_VALUE_BINS = 24
BEND_BINS = 49
TICK_CHUNKS = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
)
ROOTS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
ROOT_TO_PC = {root: idx for idx, root in enumerate(ROOTS)}
ROOT_TO_PC.update({"C#": 1, "D#": 3, "F#": 6, "G#": 8, "A#": 10})
CHORD_TYPES = (
    "maj",
    "min",
    "7",
    "maj7",
    "min7",
    "dim",
    "aug",
    "sus",
    "hdim",
    "alt",
    "other",
)
CHORD_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(CHORD_TYPES)}


def _char(token_id: int) -> str:
    return chr(PUA_BASE + token_id)


def _token_id(ch: str) -> int | None:
    if len(ch) != 1:
        return None
    value = ord(ch) - PUA_BASE
    return value if value >= 0 else None


def control_token_chars() -> set[str]:
    return {
        *(_char(ROOT_BASE + idx) for idx in range(len(ROOTS))),
        *(_char(CHORD_BASE + idx) for idx in range(len(CHORD_TYPES))),
    }


def base_token_chars(include_controls: bool = True) -> List[str]:
    chars = [
        *(_char(token_id) for token_id in (START, END, NOTE, CC, BEND, PROGRAM)),
        *(_char(PITCH_BASE + idx) for idx in range(128)),
        *(_char(VELOCITY_BASE + idx) for idx in range(VELOCITY_BINS)),
        *(_char(CC_NUMBER_BASE + idx) for idx in range(128)),
        *(_char(CC_VALUE_BASE + idx) for idx in range(CC_VALUE_BINS)),
        *(_char(BEND_BASE + idx) for idx in range(BEND_BINS)),
        *(_char(PROGRAM_BASE + idx) for idx in range(128)),
        *(_char(TIME_BASE + idx) for idx in range(len(TICK_CHUNKS))),
        *(_char(DURATION_BASE + idx) for idx in range(len(TICK_CHUNKS))),
    ]
    if include_controls:
        chars.extend(sorted(control_token_chars()))
    return chars


def base_vocabulary_seed_texts(include_controls: bool = True) -> List[str]:
    return base_token_chars(include_controls=include_controls)


def is_control_token_char(ch: str) -> bool:
    return ch in control_token_chars()


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _quantize(value: int, lo: int, hi: int, bins: int) -> int:
    value = _clamp(value, lo, hi)
    if bins <= 1:
        return 0
    return int(round(((value - lo) / (hi - lo)) * (bins - 1)))


def _dequantize(bin_idx: int, lo: int, hi: int, bins: int) -> int:
    bin_idx = _clamp(bin_idx, 0, bins - 1)
    if bins <= 1:
        return lo
    return _clamp(int(round(lo + (bin_idx / (bins - 1)) * (hi - lo))), lo, hi)


def velocity_to_bin(value: int) -> int:
    return _quantize(value, 1, 127, VELOCITY_BINS)


def bin_to_velocity(bin_idx: int) -> int:
    return _dequantize(bin_idx, 1, 127, VELOCITY_BINS)


def cc_value_to_bin(value: int) -> int:
    return _quantize(value, 0, 127, CC_VALUE_BINS)


def bin_to_cc_value(bin_idx: int) -> int:
    return _dequantize(bin_idx, 0, 127, CC_VALUE_BINS)


def bend_to_bin(value: int) -> int:
    return _quantize(value, -8192, 8191, BEND_BINS)


def bin_to_bend(bin_idx: int) -> int:
    return _dequantize(bin_idx, -8192, 8191, BEND_BINS)


def _encode_int_chunks(value: int, base: int) -> List[str]:
    value = max(0, int(value))
    tokens: List[str] = []
    for idx in reversed(range(len(TICK_CHUNKS))):
        chunk = TICK_CHUNKS[idx]
        while value >= chunk:
            tokens.append(_char(base + idx))
            value -= chunk
    return tokens


def _chunk_value(token_id: int, base: int) -> int | None:
    idx = token_id - base
    if 0 <= idx < len(TICK_CHUNKS):
        return TICK_CHUNKS[idx]
    return None


def _pitch_char(pitch: int) -> str:
    return _char(PITCH_BASE + _clamp(pitch, 0, 127))


def _velocity_char(velocity: int) -> str:
    return _char(VELOCITY_BASE + velocity_to_bin(velocity))


def _cc_number_char(control: int) -> str:
    return _char(CC_NUMBER_BASE + _clamp(control, 0, 127))


def _cc_value_char(value: int) -> str:
    return _char(CC_VALUE_BASE + cc_value_to_bin(value))


def _bend_char(value: int) -> str:
    return _char(BEND_BASE + bend_to_bin(value))


def _program_char(program: int) -> str:
    return _char(PROGRAM_BASE + _clamp(program, 0, 127))


def _is_pitch(token_id: int) -> bool:
    return PITCH_BASE <= token_id < PITCH_BASE + 128


def _is_velocity(token_id: int) -> bool:
    return VELOCITY_BASE <= token_id < VELOCITY_BASE + VELOCITY_BINS


def _is_cc_number(token_id: int) -> bool:
    return CC_NUMBER_BASE <= token_id < CC_NUMBER_BASE + 128


def _is_cc_value(token_id: int) -> bool:
    return CC_VALUE_BASE <= token_id < CC_VALUE_BASE + CC_VALUE_BINS


def _is_bend(token_id: int) -> bool:
    return BEND_BASE <= token_id < BEND_BASE + BEND_BINS


def _is_program(token_id: int) -> bool:
    return PROGRAM_BASE <= token_id < PROGRAM_BASE + 128


def _is_control(token_id: int) -> bool:
    return ROOT_BASE <= token_id < ROOT_BASE + len(ROOTS) or CHORD_BASE <= token_id < CHORD_BASE + len(CHORD_TYPES)


def _body_chars(text: str) -> List[str]:
    return [ch for ch in text if _token_id(ch) is not None]


def _encode_header(metadata: Dict[str, str]) -> str:
    ordered = {"ppq": str(metadata.get("ppq", DEFAULT_PPQ))}
    for key in sorted(metadata):
        if key == "ppq" or key.startswith("_"):
            continue
        value = str(metadata[key]).replace("\n", " ").strip()
        if value:
            ordered[key] = value.replace("|", "/").replace(" ", "_")
    return HEADER_PREFIX + "|".join(f"{key}={value}" for key, value in ordered.items())


def _parse_header(line: str) -> Dict[str, str]:
    assert line.startswith(HEADER_PREFIX), "NJamV4 document must start with 'NV4|' header."
    metadata: Dict[str, str] = {}
    for item in line.split("|")[1:]:
        assert "=" in item, f"Malformed header field: {item!r}"
        key, value = item.split("=", 1)
        metadata[key] = value
    metadata.setdefault("ppq", str(DEFAULT_PPQ))
    return metadata


def _split_document(text: str) -> tuple[Dict[str, str], str]:
    stripped = text.strip()
    assert stripped, "NJamV4 document must not be empty."
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and lines[0].startswith(HEADER_PREFIX):
        return _parse_header(lines[0]), "\n".join(lines[1:]).strip()
    return {"ppq": str(DEFAULT_PPQ)}, stripped


def body_text(text: str) -> str:
    return _split_document(text)[1]


def header_text(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith(HEADER_PREFIX):
        return ""
    return [line.strip() for line in stripped.splitlines() if line.strip()][0]


def extract_header_metadata(text: str) -> Dict[str, str]:
    return _split_document(text)[0]


def _normalize_chord_type(raw: str) -> str:
    suffix = raw.strip().lower()
    if suffix in {"-", "-m", "-min", "−", "−m", "−min"}:
        return "min"
    if suffix in {"-7", "-m7", "-min7", "−7", "−m7", "−min7"}:
        return "min7"
    if suffix in {"ø", "ø7"}:
        return "hdim"
    value = suffix.replace("-", "").replace("−", "").replace("_", "")
    if value in {"", "ma", "maj", "major", "6", "69"}:
        return "maj"
    if value in {"m", "mi", "min", "minor"}:
        return "min"
    if "maj7" in value or "ma7" in value:
        return "maj7"
    if "m7b5" in value or "mi7b5" in value or "min7b5" in value or "half" in value:
        return "hdim"
    if "min7" in value or "mi7" in value or value == "m7":
        return "min7"
    if "dim" in value:
        return "dim"
    if "aug" in value or "+" in value:
        return "aug"
    if "sus" in value:
        return "sus"
    if "alt" in value:
        return "alt"
    if "7" in value:
        return "7"
    return "other"


def _parse_chord(chord: str) -> tuple[int, str] | None:
    cleaned = chord.strip()
    if not cleaned:
        return None
    match = re.match(r"^([A-G](?:b|#)?)(.*)$", cleaned)
    if not match:
        return None
    root, suffix = match.groups()
    if root not in ROOT_TO_PC:
        return None
    return ROOT_TO_PC[root], _normalize_chord_type(suffix)


def _controls_from_metadata(metadata: Dict[str, str]) -> List[tuple[int, int, str]]:
    raw = metadata.get("_nv4_controls")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    controls = []
    for item in payload:
        try:
            time = max(0, int(item["time"]))
            root = _clamp(int(item["root"]), 0, 11)
            chord_type = str(item["type"])
        except Exception:
            continue
        if chord_type not in CHORD_TYPE_TO_INDEX:
            chord_type = "other"
        controls.append((time, root, chord_type))
    return sorted(controls)


def with_weimar_controls(document: NJamDocument, solo) -> NJamDocument:
    controls = []
    try:
        from .weimar_db import seconds_to_ticks

        beat_onsets = [beat.onset for beat in solo.beats]
        beat_ticks = [idx * int(document.metadata.get("ppq", DEFAULT_PPQ)) for idx in range(len(solo.beats))]
        ppq = int(document.metadata.get("ppq", DEFAULT_PPQ))
        seen = set()
        for beat in solo.beats:
            parsed = _parse_chord(beat.chord)
            if parsed is None:
                continue
            root, chord_type = parsed
            time = max(0, seconds_to_ticks(beat.onset, beat_onsets, beat_ticks, ppq))
            key = (time, root, chord_type)
            if key in seen:
                continue
            seen.add(key)
            controls.append({"time": time, "root": root, "type": chord_type})
    except Exception:
        controls = []
    metadata = dict(document.metadata)
    if controls:
        metadata["_nv4_controls"] = json.dumps(controls, separators=(",", ":"))
    return NJamDocument(metadata=metadata, events=list(document.events))


def _encode_control(root: int, chord_type: str) -> List[str]:
    type_idx = CHORD_TYPE_TO_INDEX.get(chord_type, CHORD_TYPE_TO_INDEX["other"])
    return [_char(ROOT_BASE + _clamp(root, 0, 11)), _char(CHORD_BASE + type_idx)]


def _encode_event(event: NJamEvent) -> List[str]:
    if isinstance(event, NoteEvent):
        return [
            _char(NOTE),
            _pitch_char(event.pitch),
            _velocity_char(event.velocity),
            *_encode_int_chunks(max(1, event.duration), DURATION_BASE),
        ]
    if isinstance(event, ControlChangeEvent):
        return [_char(CC), _cc_number_char(event.control), _cc_value_char(event.value)]
    if isinstance(event, PitchBendEvent):
        return [_char(BEND), _bend_char(event.value)]
    if isinstance(event, ProgramChangeEvent):
        return [_char(PROGRAM), _program_char(event.program)]
    raise AssertionError(f"Unsupported NJamV4 event type: {type(event)}")


def encode_document(document: NJamDocument) -> str:
    assert document.events, "NJamDocument must contain at least one event."
    tokens: List[str] = [_char(START)]
    current_time = 0
    controls = _controls_from_metadata(document.metadata)
    control_idx = 0
    timed_items = [(event.time, "event", event) for event in document.sorted_events()]
    while control_idx < len(controls):
        time, root, chord_type = controls[control_idx]
        timed_items.append((time, "control", (root, chord_type)))
        control_idx += 1
    timed_items.sort(key=lambda item: (item[0], 0 if item[1] == "control" else 1))
    for item_time, kind, payload in timed_items:
        assert item_time >= current_time, "Events must be sorted by non-decreasing time."
        tokens.extend(_encode_int_chunks(item_time - current_time, TIME_BASE))
        current_time = item_time
        if kind == "control":
            root, chord_type = payload
            tokens.extend(_encode_control(root, chord_type))
        else:
            tokens.extend(_encode_event(payload))
    tokens.append(_char(END))
    return _encode_header(document.metadata) + "\n" + "".join(tokens) + "\n"


def _consume_duration(chars: Sequence[str], idx: int) -> tuple[int, int]:
    duration = 0
    while idx < len(chars):
        token_id = _token_id(chars[idx])
        if token_id is None:
            idx += 1
            continue
        chunk = _chunk_value(token_id, DURATION_BASE)
        if chunk is None:
            break
        duration += chunk
        idx += 1
    return max(1, duration), idx


def _parse_body(body: str, metadata: Dict[str, str], strict: bool) -> NJamDocument:
    chars = _body_chars(body)
    events: List[NJamEvent] = []
    current_time = 0
    idx = 0
    while idx < len(chars):
        token_id = _token_id(chars[idx])
        idx += 1
        if token_id is None:
            continue
        chunk = _chunk_value(token_id, TIME_BASE)
        if chunk is not None:
            current_time += chunk
            continue
        if token_id in {START, END} or _is_control(token_id):
            continue
        if token_id == NOTE:
            if idx + 2 > len(chars):
                if strict:
                    raise AssertionError("Incomplete NJamV4 note event.")
                break
            pitch_id = _token_id(chars[idx])
            velocity_id = _token_id(chars[idx + 1])
            if pitch_id is None or velocity_id is None or not _is_pitch(pitch_id) or not _is_velocity(velocity_id):
                if strict:
                    raise AssertionError("Malformed NJamV4 note event.")
                continue
            idx += 2
            duration, idx = _consume_duration(chars, idx)
            events.append(
                NoteEvent(
                    time=current_time,
                    pitch=pitch_id - PITCH_BASE,
                    velocity=bin_to_velocity(velocity_id - VELOCITY_BASE),
                    duration=duration,
                )
            )
            continue
        if token_id == CC:
            if idx + 2 > len(chars):
                if strict:
                    raise AssertionError("Incomplete NJamV4 CC event.")
                break
            control_id = _token_id(chars[idx])
            value_id = _token_id(chars[idx + 1])
            idx += 2
            if control_id is None or value_id is None or not _is_cc_number(control_id) or not _is_cc_value(value_id):
                if strict:
                    raise AssertionError("Malformed NJamV4 CC event.")
                continue
            events.append(
                ControlChangeEvent(
                    time=current_time,
                    control=control_id - CC_NUMBER_BASE,
                    value=bin_to_cc_value(value_id - CC_VALUE_BASE),
                )
            )
            continue
        if token_id == BEND:
            if idx >= len(chars):
                if strict:
                    raise AssertionError("Incomplete NJamV4 pitch-bend event.")
                break
            bend_id = _token_id(chars[idx])
            idx += 1
            if bend_id is None or not _is_bend(bend_id):
                if strict:
                    raise AssertionError("Malformed NJamV4 pitch-bend event.")
                continue
            events.append(PitchBendEvent(time=current_time, value=bin_to_bend(bend_id - BEND_BASE)))
            continue
        if token_id == PROGRAM:
            if idx >= len(chars):
                if strict:
                    raise AssertionError("Incomplete NJamV4 program event.")
                break
            program_id = _token_id(chars[idx])
            idx += 1
            if program_id is None or not _is_program(program_id):
                if strict:
                    raise AssertionError("Malformed NJamV4 program event.")
                continue
            events.append(ProgramChangeEvent(time=current_time, program=program_id - PROGRAM_BASE))
            continue
        if strict:
            raise AssertionError(f"Unexpected NJamV4 token id: {token_id}")
    assert events, "NJamV4 body must contain at least one event."
    return NJamDocument(metadata=metadata, events=events)


def parse_document(text: str) -> NJamDocument:
    metadata, body = _split_document(text)
    return _parse_body(body, metadata, strict=True)


def _event_candidate_count(text: str) -> int:
    candidates = 0
    for ch in _body_chars(text):
        token_id = _token_id(ch)
        if token_id in {NOTE, CC, BEND, PROGRAM}:
            candidates += 1
    return candidates


def recover_continuation_document(text: str, metadata: Dict[str, str] | None = None) -> NJamDocument | None:
    try:
        document = _parse_body(body_text(text), {"ppq": str(DEFAULT_PPQ), **(metadata or {})}, strict=False)
    except Exception:
        return None
    return document if document.events else None


def analyze_parseable_continuation(text: str) -> ContinuationRecoveryStats:
    candidates = _event_candidate_count(text)
    recovered = 0
    hard_failures = 0
    try:
        document = _parse_body(body_text(text), {"ppq": str(DEFAULT_PPQ)}, strict=False)
        recovered = len(document.events)
    except Exception:
        hard_failures = candidates
    return ContinuationRecoveryStats(
        event_candidates=candidates,
        events_recovered=recovered,
        default_injections=0,
        clamped_fields=0,
        recovered_field_count=recovered,
        hard_failures=hard_failures,
    )


def count_parseable_continuation_events(text: str) -> int:
    return analyze_parseable_continuation(text).events_recovered


def token_debug_name(ch: str) -> str:
    token_id = _token_id(ch)
    if token_id is None:
        return "unknown"
    if token_id == START:
        return "start"
    if token_id == END:
        return "end"
    if token_id == NOTE:
        return "note"
    if token_id == CC:
        return "cc"
    if token_id == BEND:
        return "bend"
    if token_id == PROGRAM:
        return "program"
    if _is_pitch(token_id):
        return f"pitch_{token_id - PITCH_BASE}"
    if _is_velocity(token_id):
        return f"velocity_bin_{token_id - VELOCITY_BASE}"
    if _is_cc_number(token_id):
        return f"cc_{token_id - CC_NUMBER_BASE}"
    if _is_cc_value(token_id):
        return f"cc_value_bin_{token_id - CC_VALUE_BASE}"
    if _is_bend(token_id):
        return f"bend_bin_{token_id - BEND_BASE}"
    if _is_program(token_id):
        return f"program_{token_id - PROGRAM_BASE}"
    chunk = _chunk_value(token_id, TIME_BASE)
    if chunk is not None:
        return f"time_{chunk}"
    chunk = _chunk_value(token_id, DURATION_BASE)
    if chunk is not None:
        return f"duration_{chunk}"
    if ROOT_BASE <= token_id < ROOT_BASE + len(ROOTS):
        return f"root_{ROOTS[token_id - ROOT_BASE]}"
    if CHORD_BASE <= token_id < CHORD_BASE + len(CHORD_TYPES):
        return f"chord_{CHORD_TYPES[token_id - CHORD_BASE]}"
    return f"token_{token_id}"


def piece_debug_names(piece: str) -> List[str]:
    return [token_debug_name(ch) for ch in piece if _token_id(ch) is not None]


__all__ = [
    "DEFAULT_PPQ",
    "HEADER_PREFIX",
    "analyze_parseable_continuation",
    "bin_to_bend",
    "bin_to_cc_value",
    "bin_to_velocity",
    "body_text",
    "base_token_chars",
    "base_vocabulary_seed_texts",
    "cc_value_to_bin",
    "control_token_chars",
    "encode_document",
    "extract_header_metadata",
    "header_text",
    "is_control_token_char",
    "parse_document",
    "piece_debug_names",
    "recover_continuation_document",
    "token_debug_name",
    "velocity_to_bin",
    "with_weimar_controls",
]
