import tempfile
import unittest
from pathlib import Path

import mido

from super_njam.midi_tools import midi_to_njam, njam_to_midi, write_midi
from super_njam.music_language import detect_language, get_language
from super_njam.njam_v2 import analyze_parseable_continuation, encode_document, parse_document
from super_njam.njam_v3 import (
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    ProgramChangeEvent,
)
from super_njam.weimar_db import load_solo


def _event_key(event):
    if isinstance(event, NoteEvent):
        return ("note", event.time, event.pitch, event.velocity, event.duration)
    if isinstance(event, ControlChangeEvent):
        return ("cc", event.time, event.control, event.value)
    if isinstance(event, PitchBendEvent):
        return ("bend", event.time, event.value)
    if isinstance(event, ProgramChangeEvent):
        return ("program", event.time, event.channel, event.program)
    raise AssertionError(f"Unexpected event type: {type(event)}")


class NJamV2Tests(unittest.TestCase):
    def test_encode_parse_roundtrip_supports_notes_cc_bend_and_patch(self) -> None:
        document = NJamDocument(
            metadata={"ppq": "960", "tempo": "132.000", "sig": "3/4"},
            events=[
                ProgramChangeEvent(time=0, channel=0, program=65),
                NoteEvent(time=0, pitch=60, velocity=96, duration=480),
                ControlChangeEvent(time=120, control=11, value=100),
                PitchBendEvent(time=240, value=-512),
            ],
        )
        text = encode_document(document)
        self.assertTrue(text.startswith("NV2|"))
        self.assertIn("pc_65", text)
        parsed = parse_document(text)
        self.assertEqual(parsed.metadata["ppq"], "960")
        self.assertEqual(sorted(_event_key(event) for event in parsed.events), sorted(_event_key(event) for event in document.events))

    def test_bare_old_v2_body_uses_default_metadata(self) -> None:
        parsed = parse_document("<song_start> p_60 c_0 v_96 d_480 w_0 <song_end>")
        self.assertEqual(parsed.metadata["ppq"], "960")
        self.assertEqual(parsed.metadata["tempo"], "120.000")
        self.assertEqual(parsed.metadata["sig"], "4/4")
        self.assertEqual(len(parsed.events), 1)

    def test_detect_language_resolves_v2_header(self) -> None:
        self.assertEqual(detect_language("NV2|ppq=960\n<song_start> p_60 c_0 v_96 d_480 w_0 <song_end>").name, "njam-v2")

    def test_v2_document_renders_program_change_to_midi(self) -> None:
        document = parse_document("NV2|ppq=960|tempo=120.000|sig=4/4\n<song_start> pc_65 c_0 v_0 d_0 w_0 p_60 c_0 v_96 d_480 w_0 <song_end>")
        midi = njam_to_midi(document)
        messages = [msg for track in midi.tracks for msg in track if not msg.is_meta]
        self.assertTrue(any(msg.type == "program_change" and msg.program == 65 for msg in messages))
        self.assertTrue(any(msg.type == "note_on" and msg.note == 60 for msg in messages))

    def test_midi_to_v2_to_midi_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "input.mid"
            midi = mido.MidiFile(ticks_per_beat=960)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.Message("program_change", program=65, channel=0, time=0))
            track.append(mido.Message("note_on", note=62, velocity=90, channel=0, time=0))
            track.append(mido.Message("note_off", note=62, velocity=0, channel=0, time=480))
            midi.save(midi_path)

            language = get_language("njam-v2")
            document = language.midi_to_document(midi_path)
            text = language.encode_document(document)
            parsed = language.parse_document(text)
            roundtrip_path = Path(tmpdir) / "roundtrip.mid"
            write_midi(parsed, roundtrip_path)
            reopened = mido.MidiFile(roundtrip_path)
            messages = [msg for track in reopened.tracks for msg in track if not msg.is_meta]
            self.assertTrue(any(msg.type == "program_change" and msg.program == 65 for msg in messages))
            self.assertTrue(any(msg.type == "note_on" and msg.note == 62 for msg in messages))

    def test_weimar_solo_can_encode_as_v2_and_render_to_midi(self) -> None:
        db_path = Path("data/wjazzd.db")
        if not db_path.exists():
            self.skipTest("Weimar database is not available.")
        language = get_language("njam-v2")
        document = language.weimar_to_document(load_solo(db_path, 1), ppq=960)
        text = language.encode_document(document)
        parsed = language.parse_document(text)
        midi = njam_to_midi(parsed)
        messages = [msg for track in midi.tracks for msg in track if not msg.is_meta]
        self.assertTrue(any(msg.type == "note_on" for msg in messages))

    def test_recovery_stats_handle_malformed_generated_text(self) -> None:
        stats = analyze_parseable_continuation("p_60 v_bad d_480 w_0 pc_999 c_99 w_bad").to_dict()
        self.assertGreaterEqual(stats["event_candidates"], 2)
        self.assertGreaterEqual(stats["events_recovered"], 1)


if __name__ == "__main__":
    unittest.main()
