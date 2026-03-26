#!/usr/bin/env python3
"""Examples and CLI helpers for NJamV3 <-> MIDI conversion workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from super_njam.audio_tools import render_document_audio
from super_njam.midi_tools import midi_to_njam, write_midi
from super_njam.njam_v3 import (
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    encode_document,
    parse_document,
)
from super_njam.weimar_db import load_solo, weimar_to_njam


def summarize_document(document: NJamDocument) -> dict:
    notes = 0
    ccs = 0
    bends = 0
    for event in document.events:
        if isinstance(event, NoteEvent):
            notes += 1
        elif isinstance(event, ControlChangeEvent):
            ccs += 1
        elif isinstance(event, PitchBendEvent):
            bends += 1
    return {
        "ppq": document.ppq,
        "tempo": document.metadata.get("tempo", "120.0"),
        "sig": document.metadata.get("sig", "4/4"),
        "note_count": notes,
        "cc_count": ccs,
        "pitch_bend_count": bends,
    }


def convert_midi_to_njam_example(input_midi: Path, output_njam: Path) -> NJamDocument:
    document = midi_to_njam(input_midi)
    output_njam.parent.mkdir(parents=True, exist_ok=True)
    output_njam.write_text(encode_document(document), encoding="utf-8")
    return document


def convert_njam_to_midi_example(input_njam: Path, output_midi: Path) -> NJamDocument:
    document = parse_document(input_njam.read_text(encoding="utf-8"))
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    write_midi(document, output_midi)
    return document


def _default_soundfont() -> Optional[Path]:
    for candidate in (
        Path("soundfonts/soundfont.sf2"),
        Path("soundfonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"),
    ):
        if candidate.exists():
            return candidate
    return None


def _render_audio_if_requested(
    document: NJamDocument,
    output_path: Path,
    render_audio: bool,
    soundfont: Optional[Path],
) -> Optional[Path]:
    if not render_audio:
        return None
    chosen_soundfont = soundfont or _default_soundfont()
    if chosen_soundfont is None:
        raise RuntimeError(
            "Audio rendering requested but no soundfont was found. Supply --soundfont or place a .sf2 file in soundfonts/."
        )
    try:
        render_document_audio(document, output_path, soundfont_path=chosen_soundfont)
    except Exception as exc:
        raise RuntimeError(
            "Audio rendering failed. Run without --render-audio or install the local FluidSynth/pretty_midi dependencies needed for soundfont rendering."
        ) from exc
    return output_path


def _normalized_supported_metadata(document: NJamDocument) -> tuple:
    return (
        ("ppq", str(document.ppq)),
        ("sig", document.metadata.get("sig", "4/4")),
        ("tempo_milli_bpm", int(round(float(document.metadata.get("tempo", "120.0")) * 1000.0))),
    )


def _event_key(event) -> tuple:
    if isinstance(event, NoteEvent):
        return ("note", event.time, event.pitch, event.velocity, event.duration)
    if isinstance(event, ControlChangeEvent):
        return ("cc", event.time, event.control, event.value)
    if isinstance(event, PitchBendEvent):
        return ("bend", event.time, event.value)
    raise AssertionError(f"Unsupported event type: {type(event)}")


def roundtrip_summary(original: NJamDocument, rebuilt: NJamDocument) -> dict:
    original_summary = summarize_document(original)
    rebuilt_summary = summarize_document(rebuilt)
    return {
        "exact_supported_roundtrip": (
            _normalized_supported_metadata(original) == _normalized_supported_metadata(rebuilt)
            and sorted(_event_key(event) for event in original.events)
            == sorted(_event_key(event) for event in rebuilt.events)
        ),
        "original": original_summary,
        "rebuilt": rebuilt_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Example NJamV3 <-> MIDI conversion workflows.")
    sub = parser.add_subparsers(dest="command", required=True)

    midi_to_njam_parser = sub.add_parser("midi-to-njam", help="Convert a MIDI file to NJamV3 and print a summary.")
    midi_to_njam_parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="Input MIDI file to convert. Expected: a readable .mid or .midi file.",
    )
    midi_to_njam_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the generated NJamV3 text file.",
    )

    njam_to_midi_parser = sub.add_parser("njam-to-midi", help="Convert an NJamV3 file to MIDI and print a summary.")
    njam_to_midi_parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="Input NJamV3 text file to convert.",
    )
    njam_to_midi_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the rendered MIDI file.",
    )

    weimar_demo = sub.add_parser("weimar-demo", help="Export one Weimar solo through the full NJam -> MIDI -> NJam flow.")
    weimar_demo.add_argument(
        "--db",
        type=Path,
        default=Path("data/wjazzd.db"),
        help="Path to the Weimar SQLite database.",
    )
    weimar_demo.add_argument(
        "--melid",
        type=int,
        default=1,
        help="Solo melody id to export and round-trip. Expected: a positive melid present in the database.",
    )
    weimar_demo.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where the demo writes NJam, MIDI, round-trip NJam, and optional WAV output.",
    )
    weimar_demo.add_argument(
        "--render-audio",
        action="store_true",
        help="Render a WAV file from the generated MIDI using a soundfont if local rendering support is available.",
    )
    weimar_demo.add_argument(
        "--soundfont",
        type=Path,
        help="Optional soundfont path used when --render-audio is set. If omitted, the script tries a local default from soundfonts/.",
    )

    midi_demo = sub.add_parser("midi-demo", help="Run a MIDI file through MIDI -> NJam -> MIDI -> NJam and print a summary.")
    midi_demo.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="Input MIDI file to round-trip through NJam.",
    )
    midi_demo.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where the demo writes imported NJam, rebuilt MIDI, rebuilt NJam, and optional WAV output.",
    )
    midi_demo.add_argument(
        "--render-audio",
        action="store_true",
        help="Render a WAV file from the rebuilt NJam document using a soundfont if local rendering support is available.",
    )
    midi_demo.add_argument(
        "--soundfont",
        type=Path,
        help="Optional soundfont path used when --render-audio is set. If omitted, the script tries a local default from soundfonts/.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "midi-to-njam":
        document = convert_midi_to_njam_example(args.input_path, args.out)
        print(json.dumps({"output_njam": str(args.out), "summary": summarize_document(document)}, indent=2))
        return

    if args.command == "njam-to-midi":
        document = convert_njam_to_midi_example(args.input_path, args.out)
        print(json.dumps({"output_midi": str(args.out), "summary": summarize_document(document)}, indent=2))
        return

    if args.command == "weimar-demo":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        original = weimar_to_njam(load_solo(args.db, args.melid))
        original_njam_path = args.out_dir / f"melid_{args.melid}.njam"
        midi_path = args.out_dir / f"melid_{args.melid}.mid"
        rebuilt_njam_path = args.out_dir / f"melid_{args.melid}.roundtrip.njam"
        audio_path = args.out_dir / f"melid_{args.melid}.wav"

        original_njam_path.write_text(encode_document(original), encoding="utf-8")
        write_midi(original, midi_path)
        rebuilt = midi_to_njam(midi_path)
        rebuilt_njam_path.write_text(encode_document(rebuilt), encoding="utf-8")
        rendered_audio = _render_audio_if_requested(rebuilt, audio_path, args.render_audio, args.soundfont)
        print(
            json.dumps(
                {
                    "original_njam": str(original_njam_path),
                    "midi": str(midi_path),
                    "roundtrip_njam": str(rebuilt_njam_path),
                    "audio": str(rendered_audio) if rendered_audio else None,
                    "roundtrip": roundtrip_summary(original, rebuilt),
                },
                indent=2,
            )
        )
        return

    if args.command == "midi-demo":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        imported_njam_path = args.out_dir / f"{args.input_path.stem}.imported.njam"
        rebuilt_midi_path = args.out_dir / f"{args.input_path.stem}.rebuilt.mid"
        rebuilt_njam_path = args.out_dir / f"{args.input_path.stem}.rebuilt.njam"
        audio_path = args.out_dir / f"{args.input_path.stem}.wav"

        imported = convert_midi_to_njam_example(args.input_path, imported_njam_path)
        write_midi(imported, rebuilt_midi_path)
        rebuilt = midi_to_njam(rebuilt_midi_path)
        rebuilt_njam_path.write_text(encode_document(rebuilt), encoding="utf-8")
        rendered_audio = _render_audio_if_requested(rebuilt, audio_path, args.render_audio, args.soundfont)
        print(
            json.dumps(
                {
                    "imported_njam": str(imported_njam_path),
                    "rebuilt_midi": str(rebuilt_midi_path),
                    "rebuilt_njam": str(rebuilt_njam_path),
                    "audio": str(rendered_audio) if rendered_audio else None,
                    "roundtrip": roundtrip_summary(imported, rebuilt),
                },
                indent=2,
            )
        )
        return

    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    main()
