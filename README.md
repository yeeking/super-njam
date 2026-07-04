# Super NJam

Super NJam is a symbolic jazz-improvisation training pipeline. It converts music
from Weimar Jazz Database solos or MIDI files into NJam text corpora, trains
small llama-compatible causal language models, evaluates generated continuations,
and exports trained models to GGUF for llama.cpp experiments.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install numpy mido pretty_midi tokenizers transformers sentencepiece torch lightning tensorboard pyfluidsynth tqdm tensorboardx
```

## NJam Languages

The pipeline is language-adapter based. Use `--language` on corpus export,
MIDI conversion, training, and evaluation commands.

- `njam-v3`: default compact human-readable event text, used by older runs.
- `njam-v2`: older, more human-readable token grammar, useful for MIDI-folder corpora and comparison runs.
- `njam-v4`: structured base-token language. It stores fixed musical atoms as Unicode Private Use Area characters, then trains SentencePiece BPE over that stream. Pitches, velocity bins, CC numbers, CC value bins, bends, programs, durations, time shifts, and optional chord/root controls are atomic before BPE.

NJam-v4 is the current recommended language for new grammar-stability experiments:

```bash
.venv/bin/python python/1_language.py export-corpus \
  --language njam-v4 \
  --db data/wjazzd.db \
  --out artifacts/corpus-v4.jsonl \
  --limit 32
```

To include all-key transpositions:

```bash
.venv/bin/python python/1_language.py export-corpus \
  --language njam-v4 \
  --db data/wjazzd.db \
  --out artifacts/corpus-v4-all-keys.jsonl \
  --limit 32 \
  --permute-to-all-keys
```

To include optional Weimar chord/root conditioning tokens in NJam-v4:

```bash
.venv/bin/python python/1_language.py export-corpus \
  --language njam-v4 \
  --db data/wjazzd.db \
  --out artifacts/corpus-v4-controls.jsonl \
  --include-control-tokens
```

Control tokens are masked from loss when they appear in tokenizer pieces.

## Corpus And MIDI Conversion

Weimar corpus export:

```bash
.venv/bin/python python/1_language.py export-corpus --db data/wjazzd.db --out artifacts/corpus.jsonl --limit 32
```

MIDI folder corpus export:

```bash
.venv/bin/python python/1_language.py export-midi-corpus \
  --language njam-v4 \
  --midi-dir data/midi \
  --out artifacts/midi-corpus-v4.jsonl
```

Single MIDI conversion:

```bash
.venv/bin/python python/1_language.py midi-to-njam --language njam-v4 --in data/midi/example.mid --out outputs/example.njam
.venv/bin/python python/1_language.py njam-to-midi --in outputs/example.njam --out outputs/example.mid
```

Training uses SentencePiece. NJam-v4 uses BPE with identity normalization,
`add_dummy_prefix=False`, `max_sentencepiece_length=32`, and a required base-token
alphabet so all structural tokens remain representable.

## Training

Basic smoke:

```bash
.venv/bin/python python/3_trainer.py \
  --language njam-v4 \
  --corpus artifacts/corpus-v4.jsonl \
  --output-dir artifacts/train_smoke_v4 \
  --max-epochs 2 \
  --seq-len 128
```

Longer run:

```bash
.venv/bin/python python/3_trainer.py \
  --language njam-v4 \
  --corpus artifacts/corpus-v4-all-keys.jsonl \
  --max-epochs 20 \
  --seq-len 1024 \
  --batch-size 16 \
  --sample-limit 1 \
  --sample-every-n-epochs 5 \
  --instrument saxophone
```

Training notes:

- If `--output-dir` is omitted, a timestamped run folder is created under `artifacts/`.
- Header metadata is dropped from training windows but kept for rendering, recovery, and summaries.
- Default `--dataset-mode partial` gives one batch per solo per epoch.
- `--dataset-mode full` preserves exhaustive sliding-window epochs.
- `--dataset-mode musical-partial` builds beat/bar-aware candidate windows, filters sparse windows with `--min-window-notes`, and keeps normal PyTorch one-item-per-`__getitem__` semantics.
- Validation supports `partial-random`, `partial`, `full`, `musical-partial`, and `musical-partial-random`.
- Validation preflight is on by default to catch validation-time OOM early.
- Early stopping defaults to `val_loss` patience `3`.
- Optimizer defaults use gradient accumulation `16`, weight decay `0.01`, max grad norm `1.0`, warmup steps `2000`, and a cosine LR schedule. Restore older behavior with `--gradient-accumulation-steps 1 --weight-decay 0 --max-grad-norm 0 --warmup-steps 0 --lr-scheduler constant`.

## Musical Evaluation

Training runs synthetic musical tests at validation epoch end by default. The
light suite prompts the model with scale and rhythm fragments, then logs:

- `musical_eval/overall`
- `musical_eval/complete_note_pattern_rate`
- `musical_eval/parseable_note_rate`
- `musical_eval/harmony_scale_adherence`
- `musical_eval/harmony_prompt_pitch_coverage`
- `musical_eval/rhythm_ioi_similarity`
- `musical_eval/rhythm_alignment`

Standalone evaluation:

```bash
.venv/bin/python python/8_musical_eval.py \
  --model-dir artifacts/train_smoke_v4 \
  --max-new-tokens 192 \
  --json-out artifacts/train_smoke_v4/musical_eval.json
```

For NJam-v4, Python generation uses grammar-constrained decoding by default.
Disable it with `--no-grammar-constrained-generation` if you want to inspect raw
model behavior. llama.cpp/GGUF generation does not yet apply the Python grammar
mask.

## GGUF Export And C++ Inference

```bash
.venv/bin/python python/5_exporter.py \
  --ckpt artifacts/train_smoke_v4/checkpoints/best.ckpt \
  --output-dir artifacts/gguf \
  --outfile model-f16.gguf \
  --outtype f16
```

The exporter infers the run folder from the checkpoint path and rebuilds the HF
model before calling llama.cpp conversion. It expects `llama.cpp` under
`libs/llama.cpp`.

C++ smoke build:

```bash
cmake -S cplusplus/llamacpp-minimal-example -B cplusplus/llamacpp-minimal-example/build
cmake --build cplusplus/llamacpp-minimal-example/build
```

C++ inference:

```bash
./cplusplus/llamacpp-minimal-example/build/super-njam-cli \
  -m artifacts/gguf/model-f16.gguf \
  -p sample_prompt.njam \
  -n 64 \
  -o sample_output.njam
```

## Training Outputs

- `train_summary.json`: resolved config, split/window stats, tokenizer info, optimizer settings, and dataset build stats.
- `checkpoints/best.ckpt` and `checkpoints/last.ckpt`: Lightning checkpoints.
- `hf_model/`: Hugging Face model plus tokenizer for Python inference/export.
- `tensorboard/`: scalar logs, sample text, and optional audio previews.
- `reference.*`: one-time held-out target renders.
- `generated_model_only.*`: continuation-only renders.
- `musical_eval_epoch_XXXX.json`: detailed synthetic musical evaluation results.

If strict NJam parsing fails during sample rendering, the renderer tries to
recover and render whatever valid continuation events it can.
