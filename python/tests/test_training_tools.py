import tempfile
import unittest
from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM

from super_njam.training_tools import (
    NJamLightningModule,
    SentencePieceTokenizerAdapter,
    SoloSlidingWindowDataset,
    TrainConfig,
    build_sentencepiece_tokenizer,
    njam_body_text,
    njam_header_text,
)


class SlidingWindowDatasetTests(unittest.TestCase):
    def _build_tokenizer(self, texts):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return build_sentencepiece_tokenizer(texts, Path(tmpdir.name), vocab_size=512)

    def test_header_tokens_are_excluded_from_training_body(self) -> None:
        text = "NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y,3C,11 T1 C1,2O\n"
        self.assertEqual(njam_header_text(text), "NV3|ppq=96|tempo=120|sig=4/4")
        self.assertEqual(njam_body_text(text), "T0 N1Y,3C,11 T1 C1,2O")

    def test_dataset_never_crosses_solo_boundaries(self) -> None:
        texts = [
            "NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y,3C,11 T1 N20,3C,11\n",
            "NV3|ppq=96|tempo=120|sig=4/4\nT0 N30,3C,11 T1 N31,3C,11\n",
        ]
        tokenizer = self._build_tokenizer([njam_body_text(text) for text in texts])
        dataset = SoloSlidingWindowDataset(texts, tokenizer, seq_len=8)
        self.assertEqual(len(dataset.window_counts_per_solo), 2)
        expected = []
        for text in texts:
            token_ids = [tokenizer.bos_token_id] + tokenizer.encode(njam_body_text(text), add_special_tokens=False) + [tokenizer.eos_token_id]
            expected.append(len(token_ids) - 1)
        self.assertEqual(dataset.window_counts_per_solo, expected)

    def test_left_padding_masks_pad_loss_at_start(self) -> None:
        texts = ["NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y,3C,11\n"]
        tokenizer = self._build_tokenizer([njam_body_text(texts[0])])
        dataset = SoloSlidingWindowDataset(texts, tokenizer, seq_len=8)
        first = dataset[0]
        masked = int((first["labels"] == -100).sum().item())
        self.assertGreater(masked, 0)
        self.assertEqual(int(first["labels"][-1].item()), tokenizer.encode(njam_body_text(texts[0]), add_special_tokens=False)[0])

    def test_all_tail_positions_are_included(self) -> None:
        text = "NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y,3C,11 T1 N20,3C,11 T1 N21,3C,11\n"
        tokenizer = self._build_tokenizer([njam_body_text(text)])
        token_ids = [tokenizer.bos_token_id] + tokenizer.encode(njam_body_text(text), add_special_tokens=False) + [tokenizer.eos_token_id]
        dataset = SoloSlidingWindowDataset([text], tokenizer, seq_len=6)
        self.assertEqual(len(dataset), len(token_ids) - 1)
        last = dataset[len(dataset) - 1]
        self.assertEqual(int(last["labels"][-1].item()), tokenizer.eos_token_id)

    def test_bos_eos_boundaries_are_preserved_per_solo(self) -> None:
        text = "NV3|ppq=96|tempo=120|sig=4/4\nT0 N1Y,3C,11\n"
        tokenizer = self._build_tokenizer([njam_body_text(text)])
        dataset = SoloSlidingWindowDataset([text], tokenizer, seq_len=4)
        first = dataset[0]
        last = dataset[len(dataset) - 1]
        self.assertIn(tokenizer.bos_token_id, first["input_ids"].tolist())
        self.assertEqual(int(last["labels"][-1].item()), tokenizer.eos_token_id)

    def test_prompt_truncation_keeps_nonempty_njam_tokens(self) -> None:
        text = (
            "NV3|ppq=96|tempo=120|sig=4/4\n"
            "T0 N1Y,3C,11 T1 C1,2O T2 B0 T3 N20,3D,10 T4 N21,3D,10 T5 C1,2P\n"
        )
        tokenizer = self._build_tokenizer([njam_body_text(text)])
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_hidden_layers=2,
                max_position_embeddings=16,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            module = NJamLightningModule(
                model=model,
                tokenizer=tokenizer,
                val_samples=[],
                config=TrainConfig(corpus_path=Path("/tmp/unused.jsonl"), output_dir=Path(tmpdir), seq_len=8),
            )
            prompt = njam_body_text(text)
            truncated = module._truncate_prompt_to_context_budget(prompt, reserved_new_tokens=8)
            self.assertTrue(truncated)
            self.assertIn("T", truncated)
            self.assertLessEqual(
                len(tokenizer.encode(truncated, add_special_tokens=False)),
                model.config.max_position_embeddings - 8 - 1,
            )


if __name__ == "__main__":
    unittest.main()
