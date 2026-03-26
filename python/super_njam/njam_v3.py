"""NJamV3 compact symbolic music format.

NJamV3 uses beat-relative timing and compact event tokens.

Header format:
    NV3|ppq=96|tempo=180.0|sig=4/4|melid=1|instrument=as|performer=Art_Pepper

Body format:
    T0 N1T,2S,9 T3 C1,2S T0 B0 T4 ...

Token categories:
    T<delta>          advance time in ticks from previous event time
    N<p>,<v>,<d>      note-on with pitch, velocity, duration
    B<v>              absolute pitch-bend value in MIDI range [-8192, 8191]
    C<c>,<v>          control change number and value

All integer payloads are base36 for compactness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Union

from .base36 import from_base36, to_base36

DEFAULT_PPQ = 96
DEFAULT_NOTE_VELOCITY = 96
DEFAULT_NOTE_DURATION = 24


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class NoteEvent:
    time: int
    pitch: int
    velocity: int
    duration: int


@dataclass(frozen=True)
class PitchBendEvent:
    time: int
    value: int


@dataclass(frozen=True)
class ControlChangeEvent:
    time: int
    control: int
    value: int


NJamEvent = Union[NoteEvent, PitchBendEvent, ControlChangeEvent]


@dataclass
class NJamDocument:
    metadata: Dict[str, str] = field(default_factory=dict)
    events: List[NJamEvent] = field(default_factory=list)

    @property
    def ppq(self) -> int:
        return int(self.metadata.get("ppq", DEFAULT_PPQ))

    def sorted_events(self) -> List[NJamEvent]:
        def key(event: NJamEvent) -> tuple:
            if isinstance(event, NoteEvent):
                rank = 2
            elif isinstance(event, ControlChangeEvent):
                rank = 1
            else:
                rank = 0
            return (event.time, rank)

        return sorted(self.events, key=key)


def _encode_header(metadata: Dict[str, str]) -> str:
    ordered = {"ppq": str(metadata.get("ppq", DEFAULT_PPQ))}
    for key in sorted(metadata):
        if key == "ppq":
            continue
        value = str(metadata[key]).replace("\n", " ").strip()
        if value:
            ordered[key] = value.replace("|", "/").replace(" ", "_")
    return "NV3|" + "|".join(f"{key}={value}" for key, value in ordered.items())


def encode_document(document: NJamDocument) -> str:
    assert document.events, "NJamDocument must contain at least one event."
    body_tokens: List[str] = []
    current_time = 0
    for event in document.sorted_events():
        assert event.time >= current_time, "Events must be sorted by non-decreasing time."
        delta = event.time - current_time
        body_tokens.append("T" + to_base36(delta))
        current_time = event.time
        if isinstance(event, NoteEvent):
            assert 0 <= event.pitch <= 127, f"Invalid MIDI pitch {event.pitch}"
            assert 1 <= event.velocity <= 127, f"Invalid velocity {event.velocity}"
            assert event.duration > 0, f"Invalid note duration {event.duration}"
            body_tokens.append(
                "N"
                + ",".join(
                    [
                        to_base36(event.pitch),
                        to_base36(event.velocity),
                        to_base36(event.duration),
                    ]
                )
            )
        elif isinstance(event, PitchBendEvent):
            assert -8192 <= event.value <= 8191, f"Invalid pitch-bend value {event.value}"
            body_tokens.append("B" + to_base36(event.value))
        elif isinstance(event, ControlChangeEvent):
            assert 0 <= event.control <= 127, f"Invalid CC number {event.control}"
            assert 0 <= event.value <= 127, f"Invalid CC value {event.value}"
            body_tokens.append("C" + ",".join([to_base36(event.control), to_base36(event.value)]))
        else:
            raise AssertionError(f"Unsupported NJam event type: {type(event)}")
    return _encode_header(document.metadata) + "\n" + " ".join(body_tokens) + "\n"


def _parse_header(line: str) -> Dict[str, str]:
    assert line.startswith("NV3|"), "NJamV3 document must start with 'NV3|' header."
    metadata: Dict[str, str] = {}
    for item in line.split("|")[1:]:
        assert "=" in item, f"Malformed header field: {item!r}"
        key, value = item.split("=", 1)
        metadata[key] = value
    if "ppq" not in metadata:
        metadata["ppq"] = str(DEFAULT_PPQ)
    return metadata


def _split_document_lines(text: str) -> List[str]:
    stripped = [line.strip() for line in text.splitlines() if line.strip()]
    if len(stripped) >= 2:
        return stripped
    assert stripped, "NJamV3 document must not be empty."
    line = stripped[0]
    assert line.startswith("NV3|"), "NJamV3 document must start with 'NV3|' header."
    body_match = re.search(r"\sT[-0-9A-Z]+", line)
    assert body_match is not None, "NJamV3 document must contain a header line and body line."
    split_at = body_match.start()
    header_line = line[:split_at].strip()
    body_line = line[split_at:].strip()
    assert body_line, "Recovered NJamV3 body is empty."
    return [header_line, body_line]


def _extract_base36_fields(payload: str) -> List[str]:
    return re.findall(r"-?[0-9A-Z]+", payload.upper())


def _parse_note_payload(payload: str, default_velocity: int, default_duration: int) -> tuple[int, int, int]:
    parts = [part.strip().upper() for part in payload.split(",")]
    if len(parts) >= 3:
        pitch_s = parts[0]
        velocity_s = parts[1] or to_base36(default_velocity)
        duration_s = parts[2] or to_base36(default_duration)
        return (
            _clamp(from_base36(pitch_s), 0, 127),
            _clamp(from_base36(velocity_s), 1, 127),
            max(1, from_base36(duration_s)),
        )
    fields = _extract_base36_fields(payload)
    assert fields, f"Malformed NJam note payload: {payload!r}"
    pitch = _clamp(from_base36(fields[0]), 0, 127)
    velocity = _clamp(from_base36(fields[1]), 1, 127) if len(fields) >= 2 else default_velocity
    duration = max(1, from_base36(fields[2])) if len(fields) >= 3 else default_duration
    return pitch, velocity, duration


def _parse_cc_payload(payload: str, default_value: int = 0) -> tuple[int, int]:
    parts = [part.strip().upper() for part in payload.split(",")]
    if len(parts) >= 2:
        control_s = parts[0]
        value_s = parts[1] or to_base36(default_value)
        return _clamp(from_base36(control_s), 0, 127), _clamp(from_base36(value_s), 0, 127)
    fields = _extract_base36_fields(payload)
    assert fields, f"Malformed NJam control payload: {payload!r}"
    control = _clamp(from_base36(fields[0]), 0, 127)
    value = _clamp(from_base36(fields[1]), 0, 127) if len(fields) >= 2 else default_value
    return control, value


def _parse_pitch_bend_payload(payload: str, default_value: int = 0) -> int:
    fields = _extract_base36_fields(payload)
    if not fields:
        return default_value
    return _clamp(from_base36(fields[0]), -8192, 8191)


def _parse_event_tokens(
    tokens: Sequence[str],
    default_note_velocity: int = DEFAULT_NOTE_VELOCITY,
    default_note_duration: int = DEFAULT_NOTE_DURATION,
) -> List[NJamEvent]:
    events: List[NJamEvent] = []
    current_time = 0
    pending_time = False
    for token in tokens:
        assert token, "Encountered empty NJam token."
        kind = token[0]
        payload = token[1:]
        if kind == "T":
            current_time += from_base36(payload)
            pending_time = True
            continue
        assert pending_time, f"Expected time token before event token {token!r}"
        if kind == "N":
            pitch, velocity, duration = _parse_note_payload(payload, default_note_velocity, default_note_duration)
            events.append(
                NoteEvent(
                    time=current_time,
                    pitch=pitch,
                    velocity=velocity,
                    duration=duration,
                )
            )
        elif kind == "B":
            events.append(PitchBendEvent(time=current_time, value=_parse_pitch_bend_payload(payload)))
        elif kind == "C":
            control, value = _parse_cc_payload(payload)
            events.append(
                ControlChangeEvent(
                    time=current_time,
                    control=control,
                    value=value,
                )
            )
        else:
            raise AssertionError(f"Unsupported NJam token {token!r}")
        pending_time = False
    return events


def parse_document(text: str) -> NJamDocument:
    stripped = _split_document_lines(text)
    header = _parse_header(stripped[0])
    tokens: List[str] = []
    for line in stripped[1:]:
        tokens.extend(line.split())
    default_note_duration = max(1, int(header.get("ppq", DEFAULT_PPQ)) // 4)
    events = _parse_event_tokens(tokens, default_note_velocity=DEFAULT_NOTE_VELOCITY, default_note_duration=default_note_duration)
    assert events, "NJamV3 body must contain at least one event."
    return NJamDocument(metadata=header, events=events)


def prompt_prefix(document: NJamDocument, ratio: float) -> str:
    assert 0.0 < ratio < 1.0, f"ratio must be in (0, 1), got {ratio}"
    events = document.sorted_events()
    prefix_count = max(1, int(len(events) * ratio))
    prefix_doc = NJamDocument(metadata=dict(document.metadata), events=events[:prefix_count])
    return encode_document(prefix_doc)
