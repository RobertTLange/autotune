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
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/autotune/nanochat_benchmark}"
STUDY_NAME="${STUDY_NAME:-nanochat_benchmark_autotune}"
STORAGE="${STORAGE:-sqlite:///$WORK_DIR/study.db}"
TRIAL_TIMEOUT_SECONDS="${NANOCHAT_TRIAL_TIMEOUT_SECONDS:-1200}"
export NANOCHAT_BENCHMARK_SEED="${NANOCHAT_BENCHMARK_SEED:-42}"
export NANOCHAT_BENCHMARK_RESULTS_DIR="${NANOCHAT_BENCHMARK_RESULTS_DIR:-$WORK_DIR/trial_results}"
NANOCHAT_DATA_IDENTITY_SHA256="$(python3 "$SCRIPT_DIR/prepare_nanochat_cache.py" verify)"
export NANOCHAT_DATA_IDENTITY_SHA256

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
