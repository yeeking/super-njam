# Super NJam

Minimal commands.

Setup:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install numpy mido pretty_midi tokenizers transformers sentencepiece torch lightning tensorboard pyfluidsynth tqdm
```

Stage 1: corpus export and NJam/MIDI conversion:

```bash
.venv/bin/python python/1_language.py export-corpus --db data/wjazzd.db --out artifacts/corpus.jsonl --limit 32
.venv/bin/python python/7_midi_and_njam.py midi-demo --in "data/midi/ArtPepper_Anthropology_FINAL.mid" --out-dir outputs --render-audio --soundfont soundfonts/soundfont.sf2
```

Stage 2: tokenizer comparison:

```bash
.venv/bin/python python/2_tokenizer.py --corpus artifacts/corpus.jsonl --out artifacts/tokenizers.json
```

Stage 3: training:

```bash
.venv/bin/python python/3_trainer.py --corpus artifacts/corpus.jsonl --output-dir artifacts/train_smoke --max-epochs 2 --seq-len 128
.venv/bin/python python/3_trainer.py --corpus artifacts/corpus.jsonl --max-epochs 2 --seq-len 128
.venv/bin/python python/3_trainer.py --corpus artifacts/corpus.jsonl --max-epochs 20 --seq-len 1024 --batch-size 16 --sample-limit 1 --sample-every-n-epochs 5 --instrument saxophone
.venv/bin/python python/5_trainer_hyper.py --corpus artifacts/corpus.jsonl --output-dir artifacts/sweep --summary-out artifacts/sweep_summary.json --max-epochs 1 --seq-len 128
```

Training notes:

- If `--output-dir` is omitted, the trainer creates a run folder under `artifacts/` from model settings and a timestamp.
- Training now uses per-solo sliding windows only. Windows never cross solo boundaries.
- Dataset windows train on NJam body/event tokens. Header metadata is still used for held-out rendering and recovery.
- Use `--sample-every-n-epochs` to reduce validation sample generation frequency.
- Validation renders default to `saxophone`.

Stage 4: GGUF export and C++ inference:

```bash
.venv/bin/python python/5_exporter.py --model-dir artifacts/train_smoke/hf_model --output-dir artifacts/gguf --outfile model-f32.gguf --outtype f32
.venv/bin/python python/6_exporter.py --model artifacts/gguf/model-f32.gguf --prompt-file sample_prompt.njam
```

C++ smoke build:

```bash
cmake -S cplusplus/llamacpp-minimal-example -B cplusplus/llamacpp-minimal-example/build
cmake --build cplusplus/llamacpp-minimal-example/build
```

C++ inference:

```bash
./cplusplus/llamacpp-minimal-example/build/super-njam-cli -m artifacts/gguf/model-f32.gguf -p sample_prompt.njam -n 64 -o sample_output.njam
```

Training output notes:

- `reference.*` files are one-time held-out target renders for comparison.
- `generated_model_only.*` files are continuation-only renders.
- If strict NJam parsing fails during training, the renderer tries to recover and render whatever valid events it can from the continuation text.
