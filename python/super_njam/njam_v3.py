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
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence, Union

from .base36 import from_base36, to_base36

DEFAULT_PPQ = 96
DEFAULT_NOTE_VELOCITY = 96
DEFAULT_NOTE_DURATION = 24


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _clamp_with_flag(value: int, lo: int, hi: int) -> tuple[int, bool]:
    clamped = _clamp(value, lo, hi)
    return clamped, clamped != value


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


@dataclass(frozen=True)
class ContinuationRecoveryStats:
    event_candidates: int = 0
    events_recovered: int = 0
    default_injections: int = 0
    clamped_fields: int = 0
    recovered_field_count: int = 0
    hard_failures: int = 0

    @property
    def field_correctness(self) -> float:
        if self.recovered_field_count <= 0:
            return 0.0
        penalty = self.default_injections + self.clamped_fields
        return max(0.0, 1.0 - (penalty / self.recovered_field_count))

    @property
    def recovery_rate(self) -> float:
        if self.event_candidates <= 0:
            return 0.0
        return self.events_recovered / self.event_candidates

    @property
    def quality_score(self) -> float:
        return self.events_recovered * self.field_correctness

    def to_dict(self) -> Dict[str, float | int]:
        return {
            **asdict(self),
            "field_correctness": self.field_correctness,
            "recovery_rate": self.recovery_rate,
            "quality_score": self.quality_score,
        }


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


def _analyze_note_payload(payload: str, default_velocity: int, default_duration: int) -> tuple[int, int, int, int, int]:
    parts = [part.strip().upper() for part in payload.split(",")]
    defaults = 0
    clamps = 0
    if len(parts) >= 3:
        pitch_s = parts[0]
        velocity_s = parts[1] or to_base36(default_velocity)
        duration_s = parts[2] or to_base36(default_duration)
        if not parts[1]:
            defaults += 1
        if not parts[2]:
            defaults += 1
        pitch_raw = from_base36(pitch_s)
        velocity_raw = from_base36(velocity_s)
        duration_raw = from_base36(duration_s)
        pitch, pitch_clamped = _clamp_with_flag(pitch_raw, 0, 127)
        velocity, velocity_clamped = _clamp_with_flag(velocity_raw, 1, 127)
        duration = max(1, duration_raw)
        if duration != duration_raw:
            clamps += 1
        clamps += int(pitch_clamped) + int(velocity_clamped)
        return pitch, velocity, duration, defaults, clamps
    fields = _extract_base36_fields(payload)
    assert fields, f"Malformed NJam note payload: {payload!r}"
    pitch_raw = from_base36(fields[0])
    pitch, pitch_clamped = _clamp_with_flag(pitch_raw, 0, 127)
    if len(fields) >= 2:
        velocity_raw = from_base36(fields[1])
        velocity, velocity_clamped = _clamp_with_flag(velocity_raw, 1, 127)
    else:
        velocity = default_velocity
        velocity_clamped = False
        defaults += 1
    if len(fields) >= 3:
        duration_raw = from_base36(fields[2])
        duration = max(1, duration_raw)
        if duration != duration_raw:
            clamps += 1
    else:
        duration = default_duration
        defaults += 1
    clamps += int(pitch_clamped) + int(velocity_clamped)
    return pitch, velocity, duration, defaults, clamps


def _parse_note_payload(payload: str, default_velocity: int, default_duration: int) -> tuple[int, int, int]:
    pitch, velocity, duration, _, _ = _analyze_note_payload(payload, default_velocity, default_duration)
    return pitch, velocity, duration


def _analyze_cc_payload(payload: str, default_value: int = 0) -> tuple[int, int, int, int]:
    parts = [part.strip().upper() for part in payload.split(",")]
    defaults = 0
    clamps = 0
    if len(parts) >= 2:
        control_s = parts[0]
        value_s = parts[1] or to_base36(default_value)
        if not parts[1]:
            defaults += 1
        control_raw = from_base36(control_s)
        value_raw = from_base36(value_s)
        control, control_clamped = _clamp_with_flag(control_raw, 0, 127)
        value, value_clamped = _clamp_with_flag(value_raw, 0, 127)
        clamps += int(control_clamped) + int(value_clamped)
        return control, value, defaults, clamps
    fields = _extract_base36_fields(payload)
    assert fields, f"Malformed NJam control payload: {payload!r}"
    control_raw = from_base36(fields[0])
    control, control_clamped = _clamp_with_flag(control_raw, 0, 127)
    if len(fields) >= 2:
        value_raw = from_base36(fields[1])
        value, value_clamped = _clamp_with_flag(value_raw, 0, 127)
    else:
        value = default_value
        value_clamped = False
        defaults += 1
    clamps += int(control_clamped) + int(value_clamped)
    return control, value, defaults, clamps


def _parse_cc_payload(payload: str, default_value: int = 0) -> tuple[int, int]:
    control, value, _, _ = _analyze_cc_payload(payload, default_value)
    return control, value


def _analyze_pitch_bend_payload(payload: str, default_value: int = 0) -> tuple[int, int, int]:
    fields = _extract_base36_fields(payload)
    if not fields:
        return default_value, 1, 0
    value_raw = from_base36(fields[0])
    value, value_clamped = _clamp_with_flag(value_raw, -8192, 8191)
    return value, 0, int(value_clamped)


def _parse_pitch_bend_payload(payload: str, default_value: int = 0) -> int:
    value, _, _ = _analyze_pitch_bend_payload(payload, default_value)
    return value


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


def extract_header_metadata(text: str) -> Dict[str, str]:
    stripped = _split_document_lines(text)
    return _parse_header(stripped[0])


def count_parseable_continuation_events(
    text: str,
    default_note_velocity: int = DEFAULT_NOTE_VELOCITY,
    default_note_duration: int = DEFAULT_NOTE_DURATION,
) -> int:
    return analyze_parseable_continuation(text, default_note_velocity, default_note_duration).events_recovered


def analyze_parseable_continuation(
    text: str,
    default_note_velocity: int = DEFAULT_NOTE_VELOCITY,
    default_note_duration: int = DEFAULT_NOTE_DURATION,
) -> ContinuationRecoveryStats:
    count = 0
    event_candidates = 0
    default_injections = 0
    clamped_fields = 0
    recovered_field_count = 0
    hard_failures = 0
    pending_time = False
    for token in text.split():
        if not token:
            continue
        kind = token[0]
        payload = token[1:]
        try:
            if kind == "T":
                from_base36(payload)
                pending_time = True
                continue
            if not pending_time:
                continue
            if kind not in {"N", "B", "C"}:
                pending_time = False
                continue
            event_candidates += 1
            if kind == "N":
                _, _, _, defaults, clamps = _analyze_note_payload(payload, default_note_velocity, default_note_duration)
                recovered_field_count += 3
            elif kind == "B":
                _, defaults, clamps = _analyze_pitch_bend_payload(payload)
                recovered_field_count += 1
            else:
                _, _, defaults, clamps = _analyze_cc_payload(payload)
                recovered_field_count += 2
            default_injections += defaults
            clamped_fields += clamps
        except Exception:
            if pending_time and kind in {"N", "B", "C"}:
                hard_failures += 1
            pending_time = False
            continue
        count += 1
        pending_time = False
    return ContinuationRecoveryStats(
        event_candidates=event_candidates,
        events_recovered=count,
        default_injections=default_injections,
        clamped_fields=clamped_fields,
        recovered_field_count=recovered_field_count,
        hard_failures=hard_failures,
    )


def recover_continuation_document(
    text: str,
    metadata: Dict[str, str] | None = None,
    default_note_velocity: int = DEFAULT_NOTE_VELOCITY,
    default_note_duration: int | None = None,
) -> NJamDocument | None:
    resolved_metadata = dict(metadata or {})
    ppq = int(resolved_metadata.get("ppq", DEFAULT_PPQ))
    duration = default_note_duration if default_note_duration is not None else max(1, ppq // 4)
    events: List[NJamEvent] = []
    current_time = 0
    pending_time = False
    for token in text.split():
        if not token:
            continue
        kind = token[0]
        payload = token[1:]
        try:
            if kind == "T":
                current_time += from_base36(payload)
                pending_time = True
                continue
            if not pending_time:
                continue
            if kind == "N":
                pitch, velocity, note_duration, _, _ = _analyze_note_payload(payload, default_note_velocity, duration)
                events.append(NoteEvent(time=current_time, pitch=pitch, velocity=velocity, duration=note_duration))
            elif kind == "B":
                value, _, _ = _analyze_pitch_bend_payload(payload)
                events.append(PitchBendEvent(time=current_time, value=value))
            elif kind == "C":
                control, value, _, _ = _analyze_cc_payload(payload)
                events.append(ControlChangeEvent(time=current_time, control=control, value=value))
            else:
                pending_time = False
                continue
        except Exception:
            pending_time = False
            continue
        pending_time = False
    if not events:
        return None
    if "ppq" not in resolved_metadata:
        resolved_metadata["ppq"] = str(ppq)
    return NJamDocument(metadata=resolved_metadata, events=events)


def prompt_prefix(document: NJamDocument, ratio: float) -> str:
    assert 0.0 < ratio < 1.0, f"ratio must be in (0, 1), got {ratio}"
    events = document.sorted_events()
    prefix_count = max(1, int(len(events) * ratio))
    prefix_doc = NJamDocument(metadata=dict(document.metadata), events=events[:prefix_count])
    return encode_document(prefix_doc)
