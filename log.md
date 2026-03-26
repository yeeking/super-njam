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
- Replaced the old packed global training dataset with per-solo sliding windows so training windows never cross solo boundaries.
- Sliding-window training now uses left-padding at solo starts, keeps tail positions, masks pad loss, and trains on NJam body/event tokens rather than headers.
- Added progress reporting for dataset preparation and switched the prep path to prefer a process pool, with safe fallback if the runtime blocks worker processes.
- Added validation prompt truncation to respect model context limits before sample generation.
- Tightened training sample rendering so malformed continuations can still be recovered, clamped, and rendered where possible instead of being discarded.
- Added continuation recovery metrics so summaries can report how many events were recoverable and how much default filling was needed.
- Trainer run folders can now be auto-named from model settings and a timestamp when `--output-dir` is omitted.

## Stage 4

- Added GGUF export, quantization, and llama.cpp benchmark helper scripts.
- Validated a working end-to-end C++ inference path with a larger 4-layer, 128-hidden, 256-context model exported as `f32` GGUF.
- Tightened the C++ CLI to default to CPU-safe inference settings on this machine and documented the stable `f32` export path in `README.md`.
- Re-verified the GGUF export path after the sliding-window dataset changes with a fresh smoke-trained checkpoint.

## Current Status

- The Python pipeline is in a usable state for real training runs: corpus export, tokenizer build, sliding-window training, sample rendering, GGUF export, and C++ inference have all been smoke tested.
- Early or very small models still often produce malformed NJam, which is expected, but the parser, recovery metrics, and recovery render path now make those outputs inspectable and sometimes audible.

## Next Steps

- Run larger and longer training jobs on the GPU machine.
- Compare model sizes, context lengths, and tokenizer settings with the hyperparameter sweep tools.
- Track whether larger checkpoints start producing fully valid continuation-only NJam more reliably than the tiny baseline runs.
- Continue GGUF export and inference benchmarking once stronger checkpoints are available.
