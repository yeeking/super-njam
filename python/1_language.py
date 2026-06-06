#!/usr/bin/env python3
"""Stage 1 CLI for NJam language and conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_njam.midi_tools import write_midi
from super_njam.music_language import detect_language, get_language
from super_njam.weimar_db import export_corpus_jsonl, load_solo, weimar_to_njam

LANGUAGE_CHOICES = ["njam-v3", "njam-v2", "njam-v4"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NJam language and conversion tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    solo = sub.add_parser("solo-to-njam", help="Export one Weimar solo to NJam text.")
    solo.add_argument("--language", choices=LANGUAGE_CHOICES, default="njam-v3")
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
        help="Optional output path for the NJam text file. If omitted, the encoded solo is printed to stdout.",
    )
    solo.add_argument(
        "--ppq",
        type=int,
        help="Pulses per quarter note used for beat-relative quantization. Defaults to 96 for NJamV3 and 960 for NJamV2.",
    )
    solo.add_argument(
        "--include-control-tokens",
        action="store_true",
        help="For NJamV4 Weimar exports, include optional chord/root control tokens in the body stream.",
    )

    corpus = sub.add_parser("export-corpus", help="Export multiple solos to JSONL corpus.")
    corpus.add_argument("--language", choices=LANGUAGE_CHOICES, default="njam-v3")
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
        help="Pulses per quarter note used during corpus quantization. Defaults to 96 for NJamV3 and 960 for NJamV2.",
    )
    corpus.add_argument(
        "--permute-to-all-keys",
        "--permute_to_all_keys",
        action="store_true",
        help="Augment the corpus with transposed copies at -5..-1 and +1..+6 semitones, keeping each original solo and its transpositions in the same split.",
    )
    corpus.add_argument(
        "--include-control-tokens",
        action="store_true",
        help="For NJamV4, include optional Weimar chord/root control tokens and mask them during training.",
    )

    midi_corpus = sub.add_parser("export-midi-corpus", help="Export MIDI files from a folder to JSONL NJam corpus.")
    midi_corpus.add_argument("--language", choices=LANGUAGE_CHOICES, default="njam-v3")
    midi_corpus.add_argument(
        "--midi-dir",
        type=Path,
        required=True,
        help="Folder containing .mid or .midi files to export.",
    )
    midi_corpus.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSONL file for the exported MIDI corpus.",
    )
    midi_corpus.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of MIDI files to export.",
    )

    to_midi = sub.add_parser("njam-to-midi", help="Convert NJam text file to MIDI.")
    to_midi.add_argument("--language", choices=LANGUAGE_CHOICES, help="Optional parser override for bare NJam text.")
    to_midi.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="Input NJam text file to parse and render. Expected: an existing .njam or plain text file.",
    )
    to_midi.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output MIDI path. Expected: a writable .mid file path.",
    )

    from_midi = sub.add_parser("midi-to-njam", help="Convert MIDI file to NJam.")
    from_midi.add_argument("--language", choices=LANGUAGE_CHOICES, default="njam-v3")
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
        help="Directory where the smoke test writes the NJam, MIDI, and round-trip artifacts. Expected: a writable directory path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "solo-to-njam":
        language = get_language(args.language)
        ppq = args.ppq if args.ppq is not None else (960 if language.name == "njam-v2" else 96)
        document = language.weimar_to_document(load_solo(args.db, args.melid), ppq=ppq)
        if args.include_control_tokens and language.name == "njam-v4":
            from super_njam import njam_v4

            document = njam_v4.with_weimar_controls(document, load_solo(args.db, args.melid))
        text = language.encode_document(document)
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return
    if args.command == "export-corpus":
        count = export_corpus_jsonl(
            args.db,
            args.out,
            limit=args.limit,
            ppq=args.ppq,
            permute_to_all_keys=args.permute_to_all_keys,
            language=args.language,
            include_control_tokens=args.include_control_tokens,
        )
        print(json.dumps({"output": str(args.out), "count": count}, indent=2))
        return
    if args.command == "export-midi-corpus":
        language = get_language(args.language)
        assert args.midi_dir.exists(), f"MIDI folder does not exist: {args.midi_dir}"
        midi_paths = sorted(path for path in args.midi_dir.rglob("*") if path.suffix.lower() in {".mid", ".midi"})
        if args.limit is not None:
            midi_paths = midi_paths[: args.limit]
        assert midi_paths, f"No .mid or .midi files found in {args.midi_dir}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with args.out.open("w", encoding="utf-8") as handle:
            for idx, midi_path in enumerate(midi_paths):
                document = language.midi_to_document(midi_path)
                record = {
                    "melid": f"midi:{midi_path.stem}",
                    "source": str(midi_path),
                    "title": midi_path.stem,
                    "performer": "",
                    "instrument": "",
                    "tempo": float(document.metadata.get("tempo", "120.0")),
                    "signature": document.metadata.get("sig", "4/4"),
                    "transpose_semitones": 0,
                    "language": language.name,
                    "text": language.encode_document(document),
                }
                handle.write(json.dumps(record) + "\n")
                count += 1
        print(json.dumps({"output": str(args.out), "count": count}, indent=2))
        return
    if args.command == "njam-to-midi":
        text = args.input_path.read_text()
        language = get_language(args.language) if args.language else detect_language(text)
        document = language.parse_document(text)
        write_midi(document, args.out)
        print(json.dumps({"output_midi": str(args.out)}, indent=2))
        return
    if args.command == "midi-to-njam":
        language = get_language(args.language)
        document = language.midi_to_document(args.input_path)
        text = language.encode_document(document)
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return
    if args.command == "smoke":
        language = get_language("njam-v3")
        args.workdir.mkdir(parents=True, exist_ok=True)
        document = weimar_to_njam(load_solo(args.db, args.melid))
        njam_path = args.workdir / f"melid_{args.melid}.njam"
        midi_path = args.workdir / f"melid_{args.melid}.mid"
        roundtrip_path = args.workdir / f"melid_{args.melid}.roundtrip.njam"
        njam_path.write_text(language.encode_document(document))
        write_midi(document, midi_path)
        roundtrip_path.write_text(language.encode_document(language.midi_to_document(midi_path)))
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
