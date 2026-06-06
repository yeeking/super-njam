"""NJamV2 human-readable symbolic music format.

NJamV2 stores relative-time event fields as readable tokens:

    NV2|ppq=960|tempo=120.000|sig=4/4
    <song_start> p_60 c_0 v_96 d_480 w_0 cc_11 c_0 v_100 d_0 w_120 <song_end>

The event model is shared with NJamV3. V2 additionally preserves MIDI program
changes through ``pc_`` events.
"""

from __future__ import annotations

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

DEFAULT_PPQ = 960
DEFAULT_TEMPO = "120.000"
DEFAULT_SIG = "4/4"
DEFAULT_NOTE_VELOCITY = 96
DEFAULT_NOTE_DURATION = 240
SONG_START = "<song_start>"
SONG_END = "<song_end>"


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _clamp_with_flag(value: int, lo: int, hi: int) -> tuple[int, bool]:
    clamped = _clamp(value, lo, hi)
    return clamped, clamped != value


def _safe_int(token: str, default: int) -> tuple[int, bool]:
    try:
        return int(token), False
    except Exception:
        return default, True


def _encode_header(metadata: Dict[str, str]) -> str:
    ordered = {
        "ppq": str(metadata.get("ppq", DEFAULT_PPQ)),
        "tempo": str(metadata.get("tempo", DEFAULT_TEMPO)),
        "sig": str(metadata.get("sig", DEFAULT_SIG)),
    }
    for key in sorted(metadata):
        if key in ordered:
            continue
        value = str(metadata[key]).replace("\n", " ").strip()
        if value:
            ordered[key] = value.replace("|", "/").replace(" ", "_")
    return "NV2|" + "|".join(f"{key}={value}" for key, value in ordered.items())


def _parse_header(line: str | None) -> Dict[str, str]:
    metadata = {"ppq": str(DEFAULT_PPQ), "tempo": DEFAULT_TEMPO, "sig": DEFAULT_SIG}
    if not line:
        return metadata
    assert line.startswith("NV2|"), "NJamV2 document must start with 'NV2|' header."
    for item in line.split("|")[1:]:
        assert "=" in item, f"Malformed header field: {item!r}"
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def _split_document(text: str) -> tuple[Dict[str, str], str]:
    stripped = text.strip()
    assert stripped, "NJamV2 document must not be empty."
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and lines[0].startswith("NV2|"):
        return _parse_header(lines[0]), "\n".join(lines[1:]).strip()
    return _parse_header(None), stripped


def body_text(text: str) -> str:
    return _split_document(text)[1]


def header_text(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("NV2|"):
        return ""
    return [line.strip() for line in stripped.splitlines() if line.strip()][0]


def extract_header_metadata(text: str) -> Dict[str, str]:
    return _split_document(text)[0]


def _value_token(prefix: str, value: int) -> str:
    return f"{prefix}{int(value)}"


def _event_tokens(event: NJamEvent, delta: int) -> List[str]:
    if isinstance(event, NoteEvent):
        return [
            _value_token("p_", event.pitch),
            "c_0",
            _value_token("v_", event.velocity),
            _value_token("d_", event.duration),
            _value_token("w_", delta),
        ]
    if isinstance(event, ControlChangeEvent):
        return [
            _value_token("cc_", event.control),
            "c_0",
            _value_token("v_", event.value),
            "d_0",
            _value_token("w_", delta),
        ]
    if isinstance(event, PitchBendEvent):
        return [
            _value_token("pw_", event.value),
            "c_0",
            "v_0",
            "d_0",
            _value_token("w_", delta),
        ]
    if isinstance(event, ProgramChangeEvent):
        return [
            _value_token("pc_", event.program),
            _value_token("c_", event.channel),
            "v_0",
            "d_0",
            _value_token("w_", delta),
        ]
    raise AssertionError(f"Unsupported NJamV2 event type: {type(event)}")


def encode_document(document: NJamDocument) -> str:
    assert document.events, "NJamDocument must contain at least one event."
    body_tokens: List[str] = [SONG_START]
    current_time = 0
    for event in document.sorted_events():
        assert event.time >= current_time, "Events must be sorted by non-decreasing time."
        delta = event.time - current_time
        body_tokens.extend(_event_tokens(event, delta))
        current_time = event.time
    body_tokens.append(SONG_END)
    return _encode_header(document.metadata) + "\n" + " ".join(body_tokens) + "\n"


EVENT_PREFIXES = ("p_", "cc_", "pc_", "pw_")
FIELD_PREFIXES = ("c_", "v_", "d_", "w_")


def _is_event_token(token: str) -> bool:
    return token.startswith(EVENT_PREFIXES)


def _is_ignorable_token(token: str) -> bool:
    return token in {SONG_START, SONG_END, "<eos>", "</s>", "<s>"}


def _parse_fields(tokens: Sequence[str]) -> tuple[Dict[str, int], int, int, int]:
    defaults = 0
    clamps = 0
    fields = {"c": 0, "v": 0, "d": 0, "w": 0}
    seen = set()
    for token in tokens:
        if "_" not in token:
            continue
        key, raw = token.split("_", 1)
        if key not in fields:
            continue
        value, defaulted = _safe_int(raw, fields[key])
        fields[key] = value
        seen.add(key)
        defaults += int(defaulted)
    for required in ("c", "v", "d", "w"):
        if required not in seen:
            defaults += 1
    fields["c"], clamped = _clamp_with_flag(fields["c"], 0, 15)
    clamps += int(clamped)
    fields["v"], clamped = _clamp_with_flag(fields["v"], 0, 127)
    clamps += int(clamped)
    fields["d"] = max(0, fields["d"])
    fields["w"] = max(0, fields["w"])
    return fields, defaults, clamps, len(seen)


def _parse_event(event_type_token: str, field_tokens: Sequence[str], current_time: int) -> tuple[NJamEvent, int, int, int, int]:
    fields, defaults, clamps, seen_count = _parse_fields(field_tokens)
    kind, raw_value = event_type_token.split("_", 1)
    value, defaulted = _safe_int(raw_value, 0)
    defaults += int(defaulted)
    event_time = current_time + fields["w"]
    if kind == "p":
        pitch, clamped = _clamp_with_flag(value, 0, 127)
        velocity = _clamp(fields["v"] or DEFAULT_NOTE_VELOCITY, 1, 127)
        duration = max(1, fields["d"] or DEFAULT_NOTE_DURATION)
        clamps += int(clamped)
        return NoteEvent(time=event_time, pitch=pitch, velocity=velocity, duration=duration), event_time, defaults, clamps, seen_count + 1
    if kind == "cc":
        control, clamped = _clamp_with_flag(value, 0, 127)
        clamps += int(clamped)
        return ControlChangeEvent(time=event_time, control=control, value=fields["v"]), event_time, defaults, clamps, seen_count + 1
    if kind == "pc":
        program, clamped = _clamp_with_flag(value, 0, 127)
        clamps += int(clamped)
        return ProgramChangeEvent(time=event_time, program=program, channel=fields["c"]), event_time, defaults, clamps, seen_count + 1
    if kind == "pw":
        bend, clamped = _clamp_with_flag(value, -8192, 8191)
        clamps += int(clamped)
        return PitchBendEvent(time=event_time, value=bend), event_time, defaults, clamps, seen_count + 1
    raise AssertionError(f"Unsupported NJamV2 event token: {event_type_token!r}")


def _event_groups(text: str) -> Iterable[tuple[str, List[str]]]:
    tokens = [token.strip() for token in text.split() if token.strip()]
    current_event: str | None = None
    fields: List[str] = []
    for token in tokens:
        if _is_ignorable_token(token):
            continue
        if _is_event_token(token):
            if current_event is not None:
                yield current_event, fields
            current_event = token
            fields = []
            continue
        if current_event is not None:
            fields.append(token)
    if current_event is not None:
        yield current_event, fields


def parse_document(text: str) -> NJamDocument:
    metadata, body = _split_document(text)
    events: List[NJamEvent] = []
    current_time = 0
    for event_token, field_tokens in _event_groups(body):
        event, current_time, _, _, _ = _parse_event(event_token, field_tokens, current_time)
        events.append(event)
    assert events, "NJamV2 body must contain at least one event."
    return NJamDocument(metadata=metadata, events=events)


def analyze_parseable_continuation(text: str) -> ContinuationRecoveryStats:
    event_candidates = 0
    events_recovered = 0
    default_injections = 0
    clamped_fields = 0
    recovered_field_count = 0
    hard_failures = 0
    current_time = 0
    for event_token, field_tokens in _event_groups(body_text(text)):
        event_candidates += 1
        try:
            _, current_time, defaults, clamps, recovered = _parse_event(event_token, field_tokens, current_time)
        except Exception:
            hard_failures += 1
            continue
        events_recovered += 1
        default_injections += defaults
        clamped_fields += clamps
        recovered_field_count += recovered
    return ContinuationRecoveryStats(
        event_candidates=event_candidates,
        events_recovered=events_recovered,
        default_injections=default_injections,
        clamped_fields=clamped_fields,
        recovered_field_count=recovered_field_count,
        hard_failures=hard_failures,
    )


def count_parseable_continuation_events(text: str) -> int:
    return analyze_parseable_continuation(text).events_recovered


def recover_continuation_document(text: str, metadata: Dict[str, str] | None = None) -> NJamDocument | None:
    resolved_metadata = {"ppq": str(DEFAULT_PPQ), "tempo": DEFAULT_TEMPO, "sig": DEFAULT_SIG}
    resolved_metadata.update(metadata or {})
    events: List[NJamEvent] = []
    current_time = 0
    for event_token, field_tokens in _event_groups(body_text(text)):
        try:
            event, current_time, _, _, _ = _parse_event(event_token, field_tokens, current_time)
        except Exception:
            continue
        events.append(event)
    if not events:
        return None
    return NJamDocument(metadata=resolved_metadata, events=events)


def prompt_prefix(document: NJamDocument, ratio: float) -> str:
    assert 0.0 < ratio < 1.0, f"ratio must be in (0, 1), got {ratio}"
    events = document.sorted_events()
    prefix_count = max(1, int(len(events) * ratio))
    prefix_doc = NJamDocument(metadata=dict(document.metadata), events=events[:prefix_count])
    return encode_document(prefix_doc)
