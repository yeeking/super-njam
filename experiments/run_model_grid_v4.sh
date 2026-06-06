#!/usr/bin/env bash
set -euo pipefail


# generate the corpus 
.venv/bin/python python/1_language.py export-corpus --db data/wjazzd.db --out artifacts/corpus-v4-all-keys.jsonl --language njam-v4  --permute-to-all-keys 


PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CORPUS="${CORPUS:-artifacts/corpus-v4-all-keys.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/model_grid_v4_all_keys}"
PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-${OUTPUT_ROOT}_preflight}"
# PREFLIGHT_RUNS="${PREFLIGHT_RUNS:-all}"
PREFLIGHT_RUNS="${PREFLIGHT_RUNS:-larger_l16_h512_heads16_ff2048}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQ_LEN="${SEQ_LEN:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-1}"
SAMPLE_EVERY_N_EPOCHS="${SAMPLE_EVERY_N_EPOCHS:-1}"
INSTRUMENT="${INSTRUMENT:-saxophone}"

if [[ ! -f "$CORPUS" ]]; then
  echo "Corpus not found: $CORPUS" >&2
  echo "Set CORPUS=/path/to/corpus.jsonl or export one first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$PREFLIGHT_OUTPUT_ROOT"

declare -a RUNS=(
  "larger_l16_h512_heads16_ff2048 16 512 16 2048"
  "larger_l16_h512_heads8_ff1536 16 512 8 1536"
  "tiny_l4_h128_heads4_ff256 4 128 4 256"
  "small_l6_h192_heads6_ff384 6 192 6 384"
  "small_l8_h256_heads8_ff512 8 256 8 512"
  "medium_l10_h320_heads8_ff768 10 320 8 768"
  "medium_l12_h384_heads8_ff1024 12 384 8 1024"
)

should_preflight_run() {
  local run_name="$1"
  local requested="${PREFLIGHT_RUNS//,/ }"
  local selected

  if [[ "$requested" == "all" ]]; then
    return 0
  fi
  if [[ "$requested" == "none" ]]; then
    return 1
  fi

  for selected in $requested; do
    if [[ "$selected" == "$run_name" ]]; then
      return 0
    fi
  done
  return 1
}

validate_preflight_runs() {
  local requested="${PREFLIGHT_RUNS//,/ }"
  local selected run run_name _layers _hidden _heads _ff found

  if [[ "$requested" == "all" || "$requested" == "none" ]]; then
    return
  fi

  for selected in $requested; do
    found=0
    for run in "${RUNS[@]}"; do
      read -r run_name _layers _hidden _heads _ff <<< "$run"
      if [[ "$selected" == "$run_name" ]]; then
        found=1
        break
      fi
    done
    if (( found == 0 )); then
      echo "Unknown preflight run: $selected" >&2
      echo "Known run names:" >&2
      for run in "${RUNS[@]}"; do
        read -r run_name _layers _hidden _heads _ff <<< "$run"
        echo "  - $run_name" >&2
      done
      exit 1
    fi
  done
}

validate_preflight_runs

echo "Corpus: $CORPUS"
echo "Output root: $OUTPUT_ROOT"
echo "Preflight output root: $PREFLIGHT_OUTPUT_ROOT"
echo "Preflight runs: $PREFLIGHT_RUNS"
echo "Shared settings: seq_len=$SEQ_LEN batch_size=$BATCH_SIZE lr=$LEARNING_RATE max_epochs=$MAX_EPOCHS early_stopping_patience=$EARLY_STOPPING_PATIENCE"
echo

declare -a FAILED_PREFLIGHTS=()
declare -a SKIPPED_PREFLIGHTS=()

for run in "${RUNS[@]}"; do
  read -r run_name layers hidden heads ff <<< "$run"
  output_dir="$PREFLIGHT_OUTPUT_ROOT/$run_name"

  if ! should_preflight_run "$run_name"; then
    echo "==> Skipping preflight $run_name"
    SKIPPED_PREFLIGHTS+=("$run_name")
    echo
    continue
  fi

  echo "==> Preflight $run_name"
  echo "    layers=$layers hidden=$hidden heads=$heads ff=$ff output=$output_dir"

  if "$PYTHON_BIN" python/3_trainer.py \
    --corpus "$CORPUS" \
    --output-dir "$output_dir" \
    --language njam-v4 \
    --max-epochs 0 \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --num-layers "$layers" \
    --hidden-size "$hidden" \
    --num-heads "$heads" \
    --intermediate-size "$ff" \
    --sample-limit 0 \
    --sample-every-n-epochs "$SAMPLE_EVERY_N_EPOCHS" \
    --instrument "$INSTRUMENT"; then
    echo "==> Preflight passed: $run_name"
  else
    echo "==> Preflight failed: $run_name" >&2
    FAILED_PREFLIGHTS+=("$run_name")
  fi
  echo
done

if (( ${#FAILED_PREFLIGHTS[@]} > 0 )); then
  echo "Preflight failed for ${#FAILED_PREFLIGHTS[@]} run(s):" >&2
  for run_name in "${FAILED_PREFLIGHTS[@]}"; do
    echo "  - $run_name" >&2
  done
  echo "No full training runs were started. Lower BATCH_SIZE and run this script again." >&2
  exit 1
fi

if (( ${#SKIPPED_PREFLIGHTS[@]} > 0 )); then
  echo "Skipped preflight for ${#SKIPPED_PREFLIGHTS[@]} run(s):"
  for run_name in "${SKIPPED_PREFLIGHTS[@]}"; do
    echo "  - $run_name"
  done
  echo
fi

echo "All selected preflights passed. Starting full training runs."
echo

for run in "${RUNS[@]}"; do
  read -r run_name layers hidden heads ff <<< "$run"
  output_dir="$OUTPUT_ROOT/$run_name"

  echo "==> Starting $run_name"
  echo "    layers=$layers hidden=$hidden heads=$heads ff=$ff output=$output_dir"

  "$PYTHON_BIN" python/3_trainer.py \
    --corpus "$CORPUS" \
    --output-dir "$output_dir" \
    --language njam-v4 \
    --max-epochs "$MAX_EPOCHS" \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --num-layers "$layers" \
    --hidden-size "$hidden" \
    --num-heads "$heads" \
    --intermediate-size "$ff" \
    --no-validation-preflight \
    --sample-limit "$SAMPLE_LIMIT" \
    --sample-every-n-epochs "$SAMPLE_EVERY_N_EPOCHS" \
    --instrument "$INSTRUMENT"

  echo "==> Finished $run_name"
  echo
done

echo "All grid runs finished. Outputs are under: $OUTPUT_ROOT"
