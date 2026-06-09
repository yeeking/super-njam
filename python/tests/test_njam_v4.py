import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from super_njam import njam_v4
from super_njam.generation_tools import (
    FREE,
    NOTE_DURATION,
    NOTE_DURATION_TAIL,
    NOTE_PITCH,
    NOTE_VELOCITY,
    advance_njam_v4_grammar_state,
    njam_v4_piece_is_allowed,
)
from super_njam.music_language import detect_language, get_language
from super_njam.njam_v3 import (
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    ProgramChangeEvent,
)
from super_njam.training_tools import SoloSlidingWindowDataset, build_sentencepiece_tokenizer, njam_body_text


class NJamV4Tests(unittest.TestCase):
    def _document(self) -> NJamDocument:
        return NJamDocument(
            metadata={"ppq": "96", "tempo": "132.5", "sig": "4/4"},
            events=[
                ProgramChangeEvent(time=0, program=65, channel=0),
                NoteEvent(time=0, pitch=60, velocity=96, duration=48),
                ControlChangeEvent(time=48, control=64, value=127),
                PitchBendEvent(time=96, value=1234),
                NoteEvent(time=144, pitch=72, velocity=48, duration=192),
            ],
        )

    def test_language_resolver_supports_v4(self) -> None:
        language = get_language("njam-v4")
        self.assertEqual(language.name, "njam-v4")
        self.assertIs(detect_language("NV4|ppq=96\n"), language)

    def test_encode_parse_roundtrip_for_supported_events(self) -> None:
        text = njam_v4.encode_document(self._document())
        parsed = njam_v4.parse_document(text)

        self.assertEqual(parsed.metadata["ppq"], "96")
        self.assertEqual(parsed.metadata["tempo"], "132.5")
        self.assertEqual(len(parsed.events), 5)
        self.assertEqual(parsed.events[0], ProgramChangeEvent(time=0, program=65, channel=0))
        self.assertEqual(parsed.events[1].time, 0)
        self.assertEqual(parsed.events[1].pitch, 60)
        self.assertLessEqual(abs(parsed.events[1].velocity - 96), 3)
        self.assertEqual(parsed.events[1].duration, 48)
        self.assertEqual(parsed.events[2], ControlChangeEvent(time=48, control=64, value=127))
        self.assertEqual(parsed.events[3].time, 96)
        self.assertLessEqual(abs(parsed.events[3].value - 1234), 180)
        self.assertEqual(parsed.events[4].time, 144)
        self.assertEqual(parsed.events[4].duration, 192)

    def test_encoded_body_is_compact_but_spaced_legacy_body_still_parses(self) -> None:
        text = njam_v4.encode_document(self._document())
        body = njam_v4.body_text(text)
        self.assertNotIn(" ", body)

        legacy_text = njam_v4.header_text(text) + "\n" + " ".join(body) + "\n"
        parsed = njam_v4.parse_document(legacy_text)
        self.assertEqual(len(parsed.events), len(self._document().events))

    def test_quantizers_stay_in_midi_ranges(self) -> None:
        for value in (-10, 0, 1, 63, 127, 200):
            self.assertGreaterEqual(njam_v4.bin_to_velocity(njam_v4.velocity_to_bin(value)), 1)
            self.assertLessEqual(njam_v4.bin_to_velocity(njam_v4.velocity_to_bin(value)), 127)
            self.assertGreaterEqual(njam_v4.bin_to_cc_value(njam_v4.cc_value_to_bin(value)), 0)
            self.assertLessEqual(njam_v4.bin_to_cc_value(njam_v4.cc_value_to_bin(value)), 127)

        for value in (-10000, -8192, 0, 8191, 10000):
            rebuilt = njam_v4.bin_to_bend(njam_v4.bend_to_bin(value))
            self.assertGreaterEqual(rebuilt, -8192)
            self.assertLessEqual(rebuilt, 8191)

    def test_time_shift_chunks_reconstruct_event_times_and_durations(self) -> None:
        document = NJamDocument(
            metadata={"ppq": "96"},
            events=[
                NoteEvent(time=0, pitch=60, velocity=80, duration=1),
                NoteEvent(time=12345, pitch=61, velocity=80, duration=9876),
            ],
        )
        parsed = njam_v4.parse_document(njam_v4.encode_document(document))
        self.assertEqual([event.time for event in parsed.events], [0, 12345])
        self.assertEqual([event.duration for event in parsed.events], [1, 9876])

    def test_weimar_controls_are_optional_and_encoded_when_present(self) -> None:
        document = NJamDocument(metadata={"ppq": "96"}, events=[NoteEvent(time=0, pitch=60, velocity=80, duration=24)])
        plain = njam_v4.encode_document(document)
        fake_solo = SimpleNamespace(
            beats=[
                SimpleNamespace(onset=0.0, chord="Cmaj7"),
                SimpleNamespace(onset=0.5, chord="G7"),
            ]
        )
        controlled = njam_v4.with_weimar_controls(document, fake_solo)
        controlled_text = njam_v4.encode_document(controlled)

        self.assertFalse(any(ch in njam_v4.body_text(plain) for ch in njam_v4.control_token_chars()))
        self.assertTrue(any(ch in njam_v4.body_text(controlled_text) for ch in njam_v4.control_token_chars()))
        self.assertEqual(len(njam_v4.parse_document(controlled_text).events), 1)

    def test_chord_normalization_handles_jazz_spellings(self) -> None:
        cases = {
            "C": (0, "maj"),
            "C-": (0, "min"),
            "C-7": (0, "min7"),
            "Cm": (0, "min"),
            "Cm7": (0, "min7"),
            "C7": (0, "7"),
            "Cmaj7": (0, "maj7"),
            "Cø": (0, "hdim"),
            "Cm7b5": (0, "hdim"),
        }
        for chord, expected in cases.items():
            with self.subTest(chord=chord):
                self.assertEqual(njam_v4._parse_chord(chord), expected)

    def test_chord_controls_are_conditioning_tokens_not_roundtrip_events(self) -> None:
        document = NJamDocument(metadata={"ppq": "96"}, events=[NoteEvent(time=0, pitch=60, velocity=80, duration=24)])
        fake_solo = SimpleNamespace(
            beats=[
                SimpleNamespace(onset=0.0, chord="Cmaj7"),
                SimpleNamespace(onset=0.5, chord="Cmaj7"),
            ]
        )
        controlled = njam_v4.with_weimar_controls(document, fake_solo)
        self.assertEqual(len(njam_v4._controls_from_metadata(controlled.metadata)), 2)
        parsed = njam_v4.parse_document(njam_v4.encode_document(controlled))
        self.assertEqual(len(parsed.events), len(document.events))
        self.assertEqual(parsed.events[0].time, document.events[0].time)
        self.assertEqual(parsed.events[0].pitch, document.events[0].pitch)
        self.assertEqual(parsed.events[0].duration, document.events[0].duration)

    def test_recovery_stats_handle_malformed_continuations(self) -> None:
        body = njam_v4.body_text(njam_v4.encode_document(self._document()))
        damaged = body[: len(body) // 2]
        stats = njam_v4.analyze_parseable_continuation(damaged)
        self.assertGreaterEqual(stats.event_candidates, stats.events_recovered)
        self.assertIsNotNone(njam_v4.recover_continuation_document(damaged))

    def test_v4_body_text_is_used_for_training(self) -> None:
        text = njam_v4.encode_document(self._document())
        self.assertEqual(njam_body_text(text, language="njam-v4"), njam_v4.body_text(text))
        self.assertFalse(njam_body_text(text, language="njam-v4").startswith("NV4|"))

    def test_bpe_tokenizer_masks_control_pieces_from_labels(self) -> None:
        document = self._document()
        document.metadata["_nv4_controls"] = '[{"time":0,"root":0,"type":"maj7"}]'
        language = get_language("njam-v4")
        text = language.encode_document(document)
        body = language.body_text(text)
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = build_sentencepiece_tokenizer(
                [body] * 8 + language.tokenizer_seed_texts(),
                Path(tmpdir),
                vocab_size=1024,
                model_type=language.tokenizer_model_type(),
                trainer_kwargs=language.tokenizer_train_kwargs(),
            )
            tokenizer.loss_mask_token_ids = language.loss_mask_token_ids(tokenizer)
            self.assertGreater(len(tokenizer.loss_mask_token_ids), 0)
            self.assertTrue(any(token_id in tokenizer.loss_mask_token_ids for token_id in tokenizer.encode(body)))

            dataset = SoloSlidingWindowDataset([text], tokenizer, seq_len=16, language="njam-v4")
            masked_counts = [int((dataset[idx]["labels"] == -100).sum().item()) for idx in range(len(dataset))]
            self.assertTrue(any(count > 0 for count in masked_counts))

    def test_v4_bpe_training_kwargs_use_long_compact_pieces_and_required_chars(self) -> None:
        language = get_language("njam-v4")
        kwargs = language.tokenizer_train_kwargs()
        self.assertEqual(kwargs["max_sentencepiece_length"], 32)
        self.assertFalse(kwargs["add_dummy_prefix"])
        self.assertFalse(kwargs["byte_fallback"])
        self.assertEqual(kwargs["required_chars"], "".join(njam_v4.base_token_chars(include_controls=True)))

    def test_tokenizer_seed_covers_unobserved_base_tokens(self) -> None:
        language = get_language("njam-v4")
        text = njam_v4.body_text(
            njam_v4.encode_document(
                NJamDocument(metadata={"ppq": "96"}, events=[NoteEvent(time=0, pitch=60, velocity=80, duration=24)])
            )
        )
        unseen_prompt = njam_v4.body_text(
            njam_v4.encode_document(
                NJamDocument(
                    metadata={"ppq": "96"},
                    events=[
                        ProgramChangeEvent(time=0, program=127),
                        NoteEvent(time=0, pitch=127, velocity=127, duration=24),
                        ControlChangeEvent(time=24, control=127, value=127),
                    ],
                )
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = build_sentencepiece_tokenizer(
                [text] * 8 + language.tokenizer_seed_texts(),
                Path(tmpdir),
                vocab_size=1024,
                model_type=language.tokenizer_model_type(),
                trainer_kwargs=language.tokenizer_train_kwargs(),
            )
            self.assertNotIn(tokenizer.unk_token_id, tokenizer.encode(unseen_prompt))

    def test_compact_bpe_learns_long_pieces_without_seed_adjacency_artifacts(self) -> None:
        language = get_language("njam-v4")
        body = language.body_text(language.encode_document(self._document()))
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = build_sentencepiece_tokenizer(
                [body] * 32 + language.tokenizer_seed_texts(),
                Path(tmpdir),
                vocab_size=1024,
                model_type=language.tokenizer_model_type(),
                trainer_kwargs=language.tokenizer_train_kwargs(),
            )
            pieces = [tokenizer.processor.id_to_piece(idx) for idx in range(tokenizer.vocab_size)]
            max_piece_primitives = max(len(njam_v4.piece_debug_names(piece)) for piece in pieces)
            self.assertGreater(max_piece_primitives, 8)
            for piece in pieces:
                names = njam_v4.piece_debug_names(piece)
                if any(name.startswith("root_") or name.startswith("chord_") for name in names):
                    self.assertEqual(len(names), 1)

    def test_v4_grammar_blocks_malformed_note_transitions(self) -> None:
        self.assertEqual(advance_njam_v4_grammar_state(FREE, njam_v4.NOTE), NOTE_PITCH)
        self.assertIsNone(advance_njam_v4_grammar_state(NOTE_PITCH, njam_v4.TIME_BASE))
        self.assertIsNone(advance_njam_v4_grammar_state(NOTE_PITCH, njam_v4.VELOCITY_BASE))
        self.assertEqual(advance_njam_v4_grammar_state(NOTE_PITCH, njam_v4.PITCH_BASE + 60), NOTE_VELOCITY)
        self.assertIsNone(advance_njam_v4_grammar_state(NOTE_VELOCITY, njam_v4.DURATION_BASE))
        self.assertEqual(advance_njam_v4_grammar_state(NOTE_VELOCITY, njam_v4.VELOCITY_BASE + 10), NOTE_DURATION)
        self.assertEqual(advance_njam_v4_grammar_state(NOTE_DURATION, njam_v4.DURATION_BASE + 3), NOTE_DURATION_TAIL)
        self.assertEqual(advance_njam_v4_grammar_state(NOTE_DURATION_TAIL, njam_v4.NOTE), NOTE_PITCH)

    def test_v4_grammar_validates_whole_bpe_pieces(self) -> None:
        valid_note_piece = [
            njam_v4.NOTE,
            njam_v4.PITCH_BASE + 60,
            njam_v4.VELOCITY_BASE + 10,
            njam_v4.DURATION_BASE + 7,
        ]
        self.assertTrue(njam_v4_piece_is_allowed(valid_note_piece, FREE))
        self.assertFalse(njam_v4_piece_is_allowed([njam_v4.NOTE, njam_v4.TIME_BASE], FREE))
        self.assertFalse(njam_v4_piece_is_allowed([njam_v4.NOTE, njam_v4.VELOCITY_BASE], FREE))
        self.assertFalse(njam_v4_piece_is_allowed([njam_v4.PITCH_BASE + 60], FREE))
        self.assertTrue(njam_v4_piece_is_allowed([njam_v4.TIME_BASE, njam_v4.NOTE], FREE))
        self.assertTrue(njam_v4_piece_is_allowed([njam_v4.PITCH_BASE + 60], NOTE_PITCH))
        self.assertFalse(njam_v4_piece_is_allowed([njam_v4.DURATION_BASE], NOTE_VELOCITY))


if __name__ == "__main__":
    unittest.main()
