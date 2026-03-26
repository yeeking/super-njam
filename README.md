# Super NJam

Minimal commands:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install numpy mido pretty_midi tokenizers transformers sentencepiece torch lightning tensorboard
```

```bash
.venv/bin/python python/1_language.py export-corpus --db data/wjazzd.db --out artifacts/corpus.jsonl --limit 32
.venv/bin/python python/2_tokenizer.py --corpus artifacts/corpus.jsonl --out artifacts/tokenizers.json
.venv/bin/python python/3_trainer.py --corpus artifacts/corpus.jsonl --output-dir artifacts/train_smoke --max-epochs 1 --seq-len 128
.venv/bin/python python/5_trainer_hyper.py --corpus artifacts/corpus.jsonl --output-dir artifacts/sweep --summary-out artifacts/sweep_summary.json --max-epochs 1 --seq-len 128
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
