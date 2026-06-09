import unittest

from super_njam.music_language import get_language
from super_njam.musical_eval import (
    _notes,
    _pitch_metrics,
    _rhythm_metrics,
    aggregate_results,
    get_eval_cases,
    note_pattern_metrics,
    prepare_eval_prompt_text,
)
from super_njam.njam_v3 import NoteEvent


class MusicalEvalTests(unittest.TestCase):
    def test_light_suite_prompts_are_parseable_v4_documents(self) -> None:
        language = get_language("njam-v4")
        cases = get_eval_cases("light")
        self.assertEqual(len(cases), 10)
        for case in cases:
            with self.subTest(case=case.name):
                text = language.encode_document(case.document)
                parsed = language.parse_document(text)
                self.assertGreater(len(_notes(parsed)), 0)
                self.assertFalse(language.body_text(text).startswith("NV4|"))

    def test_v4_eval_prompt_strips_terminal_end_token(self) -> None:
        from super_njam import njam_v4

        language = get_language("njam-v4")
        case = get_eval_cases("light")[0]
        prompt = prepare_eval_prompt_text(language, case.document)
        self.assertTrue(prompt.startswith(njam_v4._char(njam_v4.START)))
        self.assertFalse(prompt.endswith(njam_v4._char(njam_v4.END)))

    def test_harmony_metrics_reward_in_scale_continuations(self) -> None:
        case = next(case for case in get_eval_cases("light") if case.name == "c_major_ascending")
        generated = [
            NoteEvent(time=0, pitch=60, velocity=80, duration=24),
            NoteEvent(time=48, pitch=64, velocity=80, duration=24),
            NoteEvent(time=96, pitch=67, velocity=80, duration=24),
            NoteEvent(time=144, pitch=71, velocity=80, duration=24),
        ]
        scale_adherence, pitch_coverage, out_of_scale = _pitch_metrics(case, generated, enough_notes=True)
        self.assertEqual(scale_adherence, 1.0)
        self.assertGreater(pitch_coverage, 0.0)
        self.assertEqual(out_of_scale, 0.0)

    def test_harmony_metrics_penalize_out_of_scale_continuations(self) -> None:
        case = next(case for case in get_eval_cases("light") if case.name == "c_major_ascending")
        generated = [
            NoteEvent(time=0, pitch=61, velocity=80, duration=24),
            NoteEvent(time=48, pitch=63, velocity=80, duration=24),
            NoteEvent(time=96, pitch=66, velocity=80, duration=24),
            NoteEvent(time=144, pitch=70, velocity=80, duration=24),
        ]
        scale_adherence, pitch_coverage, out_of_scale = _pitch_metrics(case, generated, enough_notes=True)
        self.assertEqual(scale_adherence, 0.0)
        self.assertEqual(pitch_coverage, 0.0)
        self.assertEqual(out_of_scale, 1.0)

    def test_rhythm_alignment_allows_phase_shift(self) -> None:
        case = next(case for case in get_eval_cases("light") if case.name == "quarter_note_pulse")
        shifted_notes = [
            NoteEvent(time=24 + idx * 96, pitch=60, velocity=80, duration=24)
            for idx in range(8)
        ]
        ioi_similarity, alignment, phase_similarity = _rhythm_metrics(
            case,
            shifted_notes,
            ppq=96,
            enough_notes=True,
        )
        self.assertGreaterEqual(ioi_similarity, 0.95)
        self.assertGreaterEqual(alignment, 0.95)
        self.assertEqual(phase_similarity, 0.0)

    def test_too_few_notes_gate_harmony_and_rhythm_scores_to_zero(self) -> None:
        harmony_case = next(case for case in get_eval_cases("light") if case.category == "harmony")
        rhythm_case = next(case for case in get_eval_cases("light") if case.category == "rhythm")
        notes = [NoteEvent(time=0, pitch=60, velocity=80, duration=24)]
        self.assertEqual(_pitch_metrics(harmony_case, notes, enough_notes=False), (0.0, 0.0, 0.0))
        self.assertEqual(_rhythm_metrics(rhythm_case, notes, ppq=96, enough_notes=False), (0.0, 0.0, 0.0))

    def test_note_pattern_metrics_count_valid_v4_note_grammar(self) -> None:
        from super_njam import njam_v4

        valid = "".join(
            [
                njam_v4._char(njam_v4.NOTE),
                njam_v4._pitch_char(60),
                njam_v4._velocity_char(88),
                *njam_v4._encode_int_chunks(48, njam_v4.DURATION_BASE),
            ]
        )
        metrics = note_pattern_metrics(valid, "njam-v4")
        self.assertEqual(metrics["note_marker_count"], 1)
        self.assertEqual(metrics["complete_note_pattern_count"], 1)
        self.assertEqual(metrics["complete_note_pattern_rate"], 1.0)

    def test_note_pattern_metrics_catch_malformed_v4_note_grammar(self) -> None:
        from super_njam import njam_v4

        malformed = "".join(
            [
                njam_v4._char(njam_v4.NOTE),
                njam_v4._encode_int_chunks(48, njam_v4.TIME_BASE)[0],
                njam_v4._char(njam_v4.NOTE),
                njam_v4._velocity_char(88),
                *njam_v4._encode_int_chunks(48, njam_v4.DURATION_BASE),
                njam_v4._char(njam_v4.NOTE),
            ]
        )
        metrics = note_pattern_metrics(malformed, "njam-v4")
        self.assertEqual(metrics["note_marker_count"], 3)
        self.assertEqual(metrics["complete_note_pattern_count"], 0)
        self.assertEqual(metrics["complete_note_pattern_rate"], 0.0)

    def test_aggregate_results_handles_empty_scores(self) -> None:
        aggregates = aggregate_results([])
        self.assertEqual(aggregates["overall"], 0.0)
        self.assertEqual(aggregates["parseable_note_rate"], 0.0)
        self.assertEqual(aggregates["complete_note_pattern_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
