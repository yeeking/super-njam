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

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Union

from .base36 import from_base36, to_base36

DEFAULT_PPQ = 96


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


def _parse_event_tokens(tokens: Sequence[str]) -> List[NJamEvent]:
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
            pitch_s, velocity_s, duration_s = payload.split(",")
            events.append(
                NoteEvent(
                    time=current_time,
                    pitch=from_base36(pitch_s),
                    velocity=from_base36(velocity_s),
                    duration=from_base36(duration_s),
                )
            )
        elif kind == "B":
            events.append(PitchBendEvent(time=current_time, value=from_base36(payload)))
        elif kind == "C":
            control_s, value_s = payload.split(",")
            events.append(
                ControlChangeEvent(
                    time=current_time,
                    control=from_base36(control_s),
                    value=from_base36(value_s),
                )
            )
        else:
            raise AssertionError(f"Unsupported NJam token {token!r}")
        pending_time = False
    return events


def parse_document(text: str) -> NJamDocument:
    stripped = [line.strip() for line in text.splitlines() if line.strip()]
    assert len(stripped) >= 2, "NJamV3 document must contain a header line and body line."
    header = _parse_header(stripped[0])
    tokens: List[str] = []
    for line in stripped[1:]:
        tokens.extend(line.split())
    events = _parse_event_tokens(tokens)
    assert events, "NJamV3 body must contain at least one event."
    return NJamDocument(metadata=header, events=events)


def prompt_prefix(document: NJamDocument, ratio: float) -> str:
    assert 0.0 < ratio < 1.0, f"ratio must be in (0, 1), got {ratio}"
    events = document.sorted_events()
    prefix_count = max(1, int(len(events) * ratio))
    prefix_doc = NJamDocument(metadata=dict(document.metadata), events=events[:prefix_count])
    return encode_document(prefix_doc)

