# Project Log

## Stage 1

- Implemented the first NJamV3 package skeleton with Weimar DB extraction, beat-relative note encoding, MIDI import/export, and language parsing.
- Added CLI entry points for single-solo export, corpus export, and smoke round trips.

## Stage 2

- Added tokenizer comparison tooling for character, whitespace, BPE, custom word-level, and SentencePiece unigram strategies.

## Stage 3

- Added a portable Lightning training pipeline around `LlamaForCausalLM`, solo-level train/val/test splitting, TensorBoard logging, and sample generation.
- Added structured sweep scaffolding for small architecture comparisons.

## Stage 4

- Added GGUF export, quantization, and llama.cpp benchmark helper scripts.
- Validated a working end-to-end C++ inference path with a larger 4-layer, 128-hidden, 256-context model exported as `f32` GGUF.
- Tightened the C++ CLI to default to CPU-safe inference settings on this machine and documented the stable `f32` export path in `README.md`.
