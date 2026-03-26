# Project Log

## Stage 1

- Implemented NJamV3 as a compact beat-relative text format with note, CC, and pitch-bend events.
- Added modular Weimar DB extraction, Weimar to NJamV3 conversion, NJamV3 to MIDI conversion, and MIDI back to NJamV3 conversion.
- Added CLI entry points for single-solo export, corpus export, MIDI/NJam conversion, and round-trip smoke tests.
- Added parser recovery for malformed early-model NJam output so partial generated text can still be converted into musical events where possible.
- Added test coverage for strict round trips, malformed NJam recovery, and recovered MIDI event generation.

## Stage 2

- Added tokenizer comparison tooling for multiple strategies and a SentencePiece-based training/export path.
- Fixed the training tokenizer path so training now uses a working SentencePiece tokenizer instead of collapsing to empty encodings.

## Stage 3

- Added a portable Lightning training pipeline around `LlamaForCausalLM`, solo-level train/val/test splitting, TensorBoard logging, and sample generation.
- Added validation-time NJam, MIDI, and audio rendering during training with saved sample artifacts per epoch.
- Training sample outputs now separate reference material from generated material, and generated outputs support full-context and model-only views where valid continuation events exist.
- Default training renders now use the larger local soundfont and a saxophone program by default, with an instrument flag on the trainer CLI.
- Verified end-to-end smoke training runs, including two-epoch runs with saved checkpoints, HF exports, and rendered sample artifacts.
- Added structured sweep scaffolding for small architecture comparisons.

## Stage 4

- Added GGUF export, quantization, and llama.cpp benchmark helper scripts.
- Validated a working end-to-end C++ inference path with a larger 4-layer, 128-hidden, 256-context model exported as `f32` GGUF.
- Tightened the C++ CLI to default to CPU-safe inference settings on this machine and documented the stable `f32` export path in `README.md`.

## Current Status

- The Python pipeline is in a usable state for real training runs: corpus export, tokenizer build, model training, sample rendering, GGUF export, and C++ inference have all been smoke tested.
- Early untrained model outputs are still often malformed NJam, which is expected, but the parser and recovery path now make those outputs inspectable and in some cases renderable.

## Next Steps

- Run larger and longer training jobs on the GPU machine.
- Compare model sizes, context lengths, and tokenizer settings with the hyperparameter sweep tools.
- Once there are stronger checkpoints, validate generated model-only continuations more closely and continue GGUF export and inference benchmarking.
