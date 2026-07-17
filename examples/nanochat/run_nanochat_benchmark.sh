#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${AUTORESEARCH_DIR:-}" ]]; then
  echo "AUTORESEARCH_DIR must point at the pinned karpathy/autoresearch checkout" >&2
  exit 2
fi

TRIALS="${TRIALS:-100}"
SAMPLER="${SAMPLER:-tpe}"
SAMPLER_SEED="${SAMPLER_SEED:-42}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-86400}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/autotune/autoresearch_comparable}"
STUDY_NAME="${STUDY_NAME:-nanochat_autoresearch_228791f_seed42}"
STORAGE="${STORAGE:-sqlite:///$WORK_DIR/study.db}"
TRIAL_TIMEOUT_SECONDS="${NANOCHAT_TRIAL_TIMEOUT_SECONDS:-1200}"
if [[ "${NANOCHAT_BENCHMARK_SEED:-42}" != "42" ]]; then
  echo "NANOCHAT_BENCHMARK_SEED must be 42 for discovery" >&2
  exit 2
fi
export NANOCHAT_BENCHMARK_SEED=42
export NANOCHAT_BENCHMARK_RESULTS_DIR="$WORK_DIR/trial_results"
NANOCHAT_DATA_IDENTITY_SHA256="$(python3 "$SCRIPT_DIR/prepare_nanochat_cache.py" verify)"
export NANOCHAT_DATA_IDENTITY_SHA256
VALIDATE_FINALISTS="${VALIDATE_FINALISTS:-1}"
FINALISTS="${FINALISTS:-1}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
VALIDATION_MAX_ATTEMPTS="${VALIDATION_MAX_ATTEMPTS:-2}"

PATH="$ROOT_DIR/.venv/bin:$PATH" node "$ROOT_DIR/dist/cli.js" run "$SCRIPT_DIR/nanochat_benchmark.py" \
  --config "$SCRIPT_DIR/nanochat_search_space.yaml" \
  --trials "$TRIALS" \
  --sampler "$SAMPLER" \
  --sampler-seed "$SAMPLER_SEED" \
  --pruner none \
  --n-jobs 1 \
  --timeout-seconds "$TRIAL_TIMEOUT_SECONDS" \
  --time-budget-seconds "$TIME_BUDGET_SECONDS" \
  --storage "$STORAGE" \
  --study-name "$STUDY_NAME" \
  --work-dir "$WORK_DIR" \
  --yes \
  --json

if [[ "$VALIDATE_FINALISTS" == "1" ]]; then
  python3 "$SCRIPT_DIR/validate_nanochat.py" \
    --result "search=$WORK_DIR/results.json" \
    --output-dir "$WORK_DIR/finalist_validation" \
    --finalists "$FINALISTS" \
    --seeds "$VALIDATION_SEEDS" \
    --timeout-seconds "$TRIAL_TIMEOUT_SECONDS" \
    --max-attempts "$VALIDATION_MAX_ATTEMPTS"
fi
