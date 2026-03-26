"""MIDI conversion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import mido

from .njam_v3 import ControlChangeEvent, NJamDocument, NoteEvent, PitchBendEvent, parse_document


def njam_to_midi(document: NJamDocument) -> mido.MidiFile:
    ticks_per_beat = document.ppq
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    meta_track = mido.MidiTrack()
    note_track = mido.MidiTrack()
    midi.tracks.append(meta_track)
    midi.tracks.append(note_track)

    tempo = float(document.metadata.get("tempo", "120.0"))
    meta_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    if "sig" in document.metadata and "/" in document.metadata["sig"]:
        numerator, denominator = document.metadata["sig"].split("/", 1)
        meta_track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=int(numerator),
                denominator=int(denominator),
                time=0,
            )
        )

    absolute_messages: List[Tuple[int, mido.Message]] = []
    for event in document.sorted_events():
        if isinstance(event, NoteEvent):
            absolute_messages.append(
                (event.time, mido.Message("note_on", note=event.pitch, velocity=event.velocity, time=0))
            )
            absolute_messages.append(
                (event.time + event.duration, mido.Message("note_off", note=event.pitch, velocity=0, time=0))
            )
        elif isinstance(event, PitchBendEvent):
            absolute_messages.append((event.time, mido.Message("pitchwheel", pitch=event.value, time=0)))
        elif isinstance(event, ControlChangeEvent):
            absolute_messages.append(
                (
                    event.time,
                    mido.Message("control_change", control=event.control, value=event.value, time=0),
                )
            )
        else:
            raise AssertionError(f"Unsupported event type: {type(event)}")

    absolute_messages.sort(key=lambda item: item[0])
    last_tick = 0
    for abs_tick, msg in absolute_messages:
        delta = abs_tick - last_tick
        assert delta >= 0, "MIDI messages must be sorted by absolute time."
        msg.time = delta
        note_track.append(msg)
        last_tick = abs_tick
    return midi


def write_midi(document: NJamDocument, output_path: Path) -> None:
    midi = njam_to_midi(document)
    midi.save(str(output_path))


def midi_to_njam(path: Path) -> NJamDocument:
    assert path.exists(), f"MIDI file does not exist: {path}"
    midi = mido.MidiFile(str(path))
    ppq = midi.ticks_per_beat
    tempo = 120.0
    sig = "4/4"
    absolute = 0
    open_notes = {}
    events = []
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.is_meta:
                if message.type == "set_tempo":
                    tempo = mido.tempo2bpm(message.tempo)
                elif message.type == "time_signature":
                    sig = f"{message.numerator}/{message.denominator}"
                continue
            if message.type == "note_on" and message.velocity > 0:
                open_notes[(message.channel, message.note)] = (absolute, message.velocity)
            elif message.type in {"note_off", "note_on"}:
                key = (message.channel, message.note)
                if key in open_notes:
                    start_tick, velocity = open_notes.pop(key)
                    duration = max(1, absolute - start_tick)
                    events.append(NoteEvent(time=start_tick, pitch=message.note, velocity=velocity, duration=duration))
            elif message.type == "pitchwheel":
                events.append(PitchBendEvent(time=absolute, value=message.pitch))
            elif message.type == "control_change":
                events.append(ControlChangeEvent(time=absolute, control=message.control, value=message.value))
    assert events, f"No note or control events found in MIDI file: {path}"
    return NJamDocument(metadata={"ppq": str(ppq), "tempo": f"{tempo:.3f}", "sig": sig}, events=events)


def njam_file_to_midi(input_path: Path, output_path: Path) -> None:
    document = parse_document(input_path.read_text())
    write_midi(document, output_path)

