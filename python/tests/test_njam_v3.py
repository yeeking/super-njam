from pathlib import Path
import tempfile
import unittest

from super_njam.midi_tools import midi_to_njam, write_midi
from super_njam.njam_v3 import (
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    encode_document,
    parse_document,
)
from super_njam.weimar_db import export_corpus_jsonl, load_solo, weimar_to_njam


class NJamRoundTripTests(unittest.TestCase):
    def test_encode_parse_roundtrip(self) -> None:
        document = NJamDocument(
            metadata={"ppq": "96", "tempo": "140", "sig": "4/4"},
            events=[
                ControlChangeEvent(time=0, control=1, value=96),
                PitchBendEvent(time=0, value=512),
                NoteEvent(time=0, pitch=60, velocity=90, duration=24),
                NoteEvent(time=12, pitch=64, velocity=80, duration=18),
            ],
        )
        text = encode_document(document)
        parsed = parse_document(text)
        self.assertEqual(parsed.metadata["ppq"], "96")
        self.assertEqual(len(parsed.events), 4)

    def test_midi_roundtrip(self) -> None:
        document = NJamDocument(
            metadata={"ppq": "96", "tempo": "120", "sig": "4/4"},
            events=[
                NoteEvent(time=0, pitch=60, velocity=100, duration=32),
                ControlChangeEvent(time=0, control=1, value=96),
                PitchBendEvent(time=8, value=256),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "test.mid"
            write_midi(document, midi_path)
            parsed = midi_to_njam(midi_path)
        self.assertTrue(any(isinstance(event, NoteEvent) for event in parsed.events))


class WeimarSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
