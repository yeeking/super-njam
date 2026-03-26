from pathlib import Path
import tempfile
import unittest

import mido

from super_njam.midi_tools import midi_to_njam, njam_to_midi, write_midi
from super_njam.njam_v3 import (
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    analyze_parseable_continuation,
    count_parseable_continuation_events,
    encode_document,
    extract_header_metadata,
    parse_document,
    recover_continuation_document,
)
from super_njam.weimar_db import export_corpus_jsonl, load_solo, weimar_to_njam


def _event_key(event):
    if isinstance(event, NoteEvent):
        return ("note", event.time, event.pitch, event.velocity, event.duration)
    if isinstance(event, ControlChangeEvent):
        return ("cc", event.time, event.control, event.value)
    if isinstance(event, PitchBendEvent):
        return ("bend", event.time, event.value)
    raise AssertionError(f"Unsupported event type in test: {type(event)}")


def _tempo_key(value: str) -> int:
    return int(round(float(value) * 1000.0))


def _normalized_supported_metadata(document: NJamDocument):
    metadata = document.metadata
    return (
        ("ppq", str(document.ppq)),
        ("sig", metadata.get("sig", "4/4")),
        ("tempo_milli_bpm", _tempo_key(metadata.get("tempo", "120.0"))),
    )


def _normalized_document(document: NJamDocument):
    return (
        _normalized_supported_metadata(document),
        tuple(sorted(_event_key(event) for event in document.events)),
    )


def _summary_counts(document: NJamDocument):
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
    return {"notes": notes, "ccs": ccs, "bends": bends}


class NJamRoundTripTests(unittest.TestCase):
    def _rich_document(self) -> NJamDocument:
        # This fixture defines the exact fidelity contract for the currently
        # supported NJam <-> MIDI intersection: note data, CC, pitch bend,
        # PPQ, tempo, and time signature.
        return NJamDocument(
            metadata={"ppq": "96", "tempo": "140.125", "sig": "7/8"},
            events=[
                ControlChangeEvent(time=0, control=11, value=92),
                PitchBendEvent(time=0, value=512),
                NoteEvent(time=0, pitch=60, velocity=90, duration=24),
                NoteEvent(time=12, pitch=64, velocity=80, duration=36),
                ControlChangeEvent(time=12, control=1, value=96),
                PitchBendEvent(time=18, value=-256),
                NoteEvent(time=18, pitch=67, velocity=70, duration=12),
                ControlChangeEvent(time=18, control=1, value=64),
                PitchBendEvent(time=30, value=0),
                ControlChangeEvent(time=36, control=11, value=88),
                NoteEvent(time=36, pitch=72, velocity=100, duration=48),
                ControlChangeEvent(time=36, control=11, value=88),
                PitchBendEvent(time=48, value=1024),
                PitchBendEvent(time=60, value=0),
            ],
        )

    def test_encode_parse_roundtrip_is_lossless_for_supported_events(self) -> None:
        document = self._rich_document()
        text = encode_document(document)
        parsed = parse_document(text)
        self.assertEqual(_normalized_document(parsed), _normalized_document(document))

    def test_njam_to_midi_to_njam_is_lossless_for_supported_events(self) -> None:
        document = self._rich_document()
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "rich.mid"
            write_midi(document, midi_path)
            parsed = midi_to_njam(midi_path)
        self.assertEqual(_normalized_document(parsed), _normalized_document(document))

    def test_malformed_njam_fails_cleanly(self) -> None:
        with self.assertRaises(AssertionError):
            parse_document("NV3|ppq=96\nN1,2,3\n")

    def test_parse_document_recovers_single_line_header_and_body(self) -> None:
        text = "NV3|ppq=96|tempo=120|sig=4/4 T0 N1Y,3C,11 T1 C1,2O T1 B0\n"
        parsed = parse_document(text)
        counts = _summary_counts(parsed)
        self.assertEqual(counts["notes"], 1)
        self.assertEqual(counts["ccs"], 1)
        self.assertEqual(counts["bends"], 1)

    def test_parse_document_fills_defaults_for_partial_event_payloads(self) -> None:
        text = "NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y T1 C1 T1 B\n"
        parsed = parse_document(text)
        normalized = _normalized_document(parsed)
        self.assertIn(("note", 0, 70, 96, 24), normalized[1])
        self.assertIn(("cc", 1, 1, 0), normalized[1])
        self.assertIn(("bend", 2, 0), normalized[1])

    def test_parse_document_clamps_recovered_values_to_midi_ranges(self) -> None:
        text = "NV3|ppq=96|tempo=120|sig=4/4\nT0 NZZ,ZZ,0 T1 CZZ,ZZ T1 BZZZ\n"
        parsed = parse_document(text)
        normalized = _normalized_document(parsed)
        self.assertIn(("note", 0, 127, 127, 1), normalized[1])
        self.assertIn(("cc", 1, 127, 127), normalized[1])
        self.assertIn(("bend", 2, 8191), normalized[1])


class RealMidiWorkflowTests(unittest.TestCase):
    def test_repo_midi_files_can_be_imported_and_re_exported(self) -> None:
        midi_paths = sorted(Path("data/midi").glob("*"))
        self.assertTrue(midi_paths, "Expected at least one sample MIDI file in data/midi.")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for midi_path in midi_paths:
                with self.subTest(midi_path=midi_path.name):
                    document = midi_to_njam(midi_path)
                    counts = _summary_counts(document)
                    self.assertGreater(counts["notes"], 0, f"Expected note events in {midi_path.name}")

                    njam_text = encode_document(document)
                    reparsed = parse_document(njam_text)
                    self.assertEqual(_normalized_document(reparsed), _normalized_document(document))

                    roundtrip_midi_path = tmpdir_path / f"{midi_path.stem}.roundtrip.mid"
                    njam_to_midi(reparsed).save(str(roundtrip_midi_path))
                    reopened = mido.MidiFile(roundtrip_midi_path)
                    self.assertGreaterEqual(len(reopened.tracks), 1)

    def test_empty_midi_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "empty.mid"
            midi = mido.MidiFile(ticks_per_beat=96)
            midi.tracks.append(mido.MidiTrack())
            midi.save(midi_path)
            with self.assertRaises(AssertionError):
                midi_to_njam(midi_path)


class ParserRecoveryMidiTests(unittest.TestCase):
    def test_extract_header_metadata_recovers_header_from_single_line_prompt(self) -> None:
        metadata = extract_header_metadata("NV3|ppq=96|tempo=120|sig=4/4 T0 N1Y,3C,11")
        self.assertEqual(metadata["ppq"], "96")
        self.assertEqual(metadata["tempo"], "120")
        self.assertEqual(metadata["sig"], "4/4")

    def test_analyze_parseable_continuation_reports_defaults_and_quality(self) -> None:
        stats = analyze_parseable_continuation("T0 N1Y T1 C1 T1 B T2 NZZ,ZZ,0").to_dict()
        self.assertEqual(stats["event_candidates"], 4)
        self.assertEqual(stats["events_recovered"], 4)
        self.assertEqual(stats["default_injections"], 4)
        self.assertEqual(stats["clamped_fields"], 3)
        self.assertEqual(stats["recovered_field_count"], 9)
        self.assertEqual(stats["hard_failures"], 0)
        self.assertAlmostEqual(stats["field_correctness"], 2.0 / 9.0)
        self.assertAlmostEqual(stats["quality_score"], 8.0 / 9.0)

    def test_analyze_parseable_continuation_counts_hard_failures(self) -> None:
        stats = analyze_parseable_continuation("T0 N- T1 C1,2O T1 B").to_dict()
        self.assertEqual(stats["event_candidates"], 3)
        self.assertEqual(stats["events_recovered"], 2)
        self.assertEqual(stats["hard_failures"], 1)

    def test_count_parseable_continuation_events_counts_valid_event_tokens(self) -> None:
        text = "garbage T0 N1Y,3C,11 junk T1 C1 T1 B TQ bad T2 N20"
        self.assertEqual(count_parseable_continuation_events(text), 4)

    def test_count_parseable_continuation_events_ignores_event_tokens_without_time(self) -> None:
        text = "N1Y,3C,11 C1,2O B0 T0 N20"
        self.assertEqual(count_parseable_continuation_events(text), 1)

    def test_recover_continuation_document_renders_partial_generated_text(self) -> None:
        document = recover_continuation_document(
            "01 T C,1 T C,1 TB C,1 TJB1,2 THB15A8888 BR T C,1 TB2BYBC T",
            metadata={"ppq": "96", "tempo": "120", "sig": "4/4"},
        )
        self.assertIsNotNone(document)
        assert document is not None
        self.assertGreaterEqual(len(document.events), 1)
        midi = njam_to_midi(document)
        non_meta = [msg for track in midi.tracks for msg in track if not msg.is_meta]
        self.assertTrue(non_meta)

    def test_render_instrument_metadata_emits_program_change(self) -> None:
        document = parse_document("NV3|ppq=96|tempo=120|sig=4/4|render_instrument=saxophone\nT0 N1Y,3C,11\n")
        midi = njam_to_midi(document)
        messages = [msg for track in midi.tracks for msg in track if not msg.is_meta]
        self.assertEqual(messages[0].type, "program_change")
        self.assertEqual(messages[0].program, 65)

    def test_recovered_single_line_document_renders_midi_events(self) -> None:
        text = "NV3|ppq=96|tempo=120|sig=4/4 T0 N1Y,3C,11 T1 C1,2O T1 B0\n"
        document = parse_document(text)
        midi = njam_to_midi(document)
        counts = {}
        for track in midi.tracks:
            for msg in track:
                key = msg.type if not msg.is_meta else f"meta:{msg.type}"
                counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts.get("note_on", 0), 1)
        self.assertEqual(counts.get("note_off", 0), 1)
        self.assertEqual(counts.get("control_change", 0), 1)
        self.assertEqual(counts.get("pitchwheel", 0), 1)

    def test_partial_note_payload_defaults_render_to_note_on_and_note_off(self) -> None:
        document = parse_document("NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y\n")
        midi = njam_to_midi(document)
        note_on = []
        note_off = []
        for track in midi.tracks:
            absolute = 0
            for msg in track:
                absolute += msg.time
                if not msg.is_meta and msg.type == "note_on":
                    note_on.append((absolute, msg.note, msg.velocity))
                elif not msg.is_meta and msg.type == "note_off":
                    note_off.append((absolute, msg.note, msg.velocity))
        self.assertEqual(note_on, [(0, 70, 96)])
        self.assertEqual(note_off, [(24, 70, 0)])

    def test_partial_control_and_bend_defaults_render_to_midi(self) -> None:
        document = parse_document("NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y T1 C1 T1 B\n")
        midi = njam_to_midi(document)
        control_changes = []
        pitch_bends = []
        for track in midi.tracks:
            absolute = 0
            for msg in track:
                absolute += msg.time
                if not msg.is_meta and msg.type == "control_change":
                    control_changes.append((absolute, msg.control, msg.value))
                elif not msg.is_meta and msg.type == "pitchwheel":
                    pitch_bends.append((absolute, msg.pitch))
        self.assertEqual(control_changes, [(1, 1, 0)])
        self.assertEqual(pitch_bends, [(2, 0)])


class WeimarRoundTripTests(unittest.TestCase):
    def test_weimar_to_njam(self) -> None:
        solo = load_solo(Path("data/wjazzd.db"), 1)
        document = weimar_to_njam(solo)
        text = encode_document(document)
        self.assertIn("NV3|", text)
        self.assertIn("N", text)

    def test_export_corpus_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "corpus.jsonl"
            count = export_corpus_jsonl(Path("data/wjazzd.db"), output_path, limit=2)
            self.assertEqual(count, 2)
            self.assertTrue(output_path.read_text().strip())

    def test_weimar_generated_midi_roundtrip_is_lossless_for_supported_events(self) -> None:
        for melid in (1, 5, 10):
            with self.subTest(melid=melid):
                original = weimar_to_njam(load_solo(Path("data/wjazzd.db"), melid))
                counts = _summary_counts(original)
                self.assertGreater(counts["notes"], 0)
                self.assertGreater(counts["ccs"] + counts["bends"], 0)

                with tempfile.TemporaryDirectory() as tmpdir:
                    midi_path = Path(tmpdir) / f"melid_{melid}.mid"
                    write_midi(original, midi_path)
                    reparsed = midi_to_njam(midi_path)

                self.assertEqual(_normalized_document(reparsed), _normalized_document(original))


if __name__ == "__main__":
    unittest.main()
