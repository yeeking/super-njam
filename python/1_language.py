#!/usr/bin/env python3
"""Stage 1 CLI for NJamV3 language and conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.midi_tools import midi_to_njam, write_midi
from super_njam.njam_v3 import encode_document, parse_document
from super_njam.weimar_db import export_corpus_jsonl, load_solo, weimar_to_njam


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NJamV3 language and conversion tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    solo = sub.add_parser("solo-to-njam", help="Export one Weimar solo to NJamV3 text.")
    solo.add_argument(
        "--db",
        type=Path,
        default=Path("data/wjazzd.db"),
        help="Path to the Weimar SQLite database. Expected: an existing .db file containing the melody and beat tables.",
    )
    solo.add_argument(
        "--melid",
        type=int,
        required=True,
        help="Solo melody id to export. Expected range: a positive integer that exists in the database, for example 1-10000+ depending on the dataset.",
    )
    solo.add_argument(
        "--out",
        type=Path,
        help="Optional output path for the NJamV3 text file. If omitted, the encoded solo is printed to stdout.",
    )
    solo.add_argument(
        "--ppq",
        type=int,
        default=96,
        help="Pulses per quarter note used for beat-relative quantization. Expected range: 24-960; 96 is a practical default.",
    )

    corpus = sub.add_parser("export-corpus", help="Export multiple solos to JSONL corpus.")
    corpus.add_argument(
        "--db",
        type=Path,
        default=Path("data/wjazzd.db"),
        help="Path to the Weimar SQLite database. Expected: an existing .db file containing source solos.",
    )
    corpus.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSONL file for the exported corpus. One encoded solo is written per line.",
    )
    corpus.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of solos to export. Expected range: positive integer; omit to export every available solo.",
    )
    corpus.add_argument(
        "--ppq",
        type=int,
        default=96,
        help="Pulses per quarter note used during corpus quantization. Expected range: 24-960; higher values preserve finer timing at the cost of longer sequences.",
    )

    to_midi = sub.add_parser("njam-to-midi", help="Convert NJamV3 text file to MIDI.")
    to_midi.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="Input NJamV3 text file to parse and render. Expected: an existing .njam or plain text file in valid NJamV3 format.",
    )
    to_midi.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output MIDI path. Expected: a writable .mid file path.",
    )

    from_midi = sub.add_parser("midi-to-njam", help="Convert MIDI file to NJamV3.")
    from_midi.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="Input MIDI file to import. Expected: an existing .mid file with note events; overlapping notes are preserved where possible.",
    )
    from_midi.add_argument(
        "--out",
        type=Path,
        help="Optional output path for the NJamV3 text. If omitted, the encoded result is printed to stdout.",
    )

    smoke = sub.add_parser("smoke", help="Round-trip smoke test for a single Weimar solo.")
    smoke.add_argument(
        "--db",
        type=Path,
        default=Path("data/wjazzd.db"),
        help="Path to the Weimar SQLite database used for the smoke test.",
    )
    smoke.add_argument(
        "--melid",
        type=int,
        default=1,
        help="Solo melody id used for the round-trip smoke test. Expected range: a positive melid present in the database.",
    )
    smoke.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/super_njam_smoke"),
        help="Directory where the smoke test writes the NJamV3, MIDI, and round-trip artifacts. Expected: a writable directory path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "solo-to-njam":
        document = weimar_to_njam(load_solo(args.db, args.melid), ppq=args.ppq)
        text = encode_document(document)
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return
    if args.command == "export-corpus":
        count = export_corpus_jsonl(args.db, args.out, limit=args.limit, ppq=args.ppq)
        print(json.dumps({"output": str(args.out), "count": count}, indent=2))
        return
    if args.command == "njam-to-midi":
        document = parse_document(args.input_path.read_text())
        write_midi(document, args.out)
        print(json.dumps({"output_midi": str(args.out)}, indent=2))
        return
    if args.command == "midi-to-njam":
        document = midi_to_njam(args.input_path)
        text = encode_document(document)
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return
    if args.command == "smoke":
        args.workdir.mkdir(parents=True, exist_ok=True)
        document = weimar_to_njam(load_solo(args.db, args.melid))
        njam_path = args.workdir / f"melid_{args.melid}.njam"
        midi_path = args.workdir / f"melid_{args.melid}.mid"
        roundtrip_path = args.workdir / f"melid_{args.melid}.roundtrip.njam"
        njam_path.write_text(encode_document(document))
        write_midi(document, midi_path)
        roundtrip_path.write_text(encode_document(midi_to_njam(midi_path)))
        print(
            json.dumps(
                {
                    "njam_path": str(njam_path),
                    "midi_path": str(midi_path),
                    "roundtrip_path": str(roundtrip_path),
                },
                indent=2,
            )
        )
        return
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    main()
