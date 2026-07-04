# Super NJam llama.cpp Smoke CLI

This is the minimal C++ path for loading a Super NJam GGUF model and generating
from an NJam prompt file with llama.cpp.

Build from the repository root:

```bash
cmake -S cplusplus/llamacpp-minimal-example -B cplusplus/llamacpp-minimal-example/build
cmake --build cplusplus/llamacpp-minimal-example/build
```

Run:

```bash
./cplusplus/llamacpp-minimal-example/build/super-njam-cli \
  -m artifacts/gguf/model-f16.gguf \
  -p sample_prompt.njam \
  -n 64 \
  -o sample_output.njam
```

The Python NJam-v4 grammar logits processor is not applied in this C++ smoke
path yet.
