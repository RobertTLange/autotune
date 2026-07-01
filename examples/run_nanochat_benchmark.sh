#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${NANOCHAT_DIR:-}" ]]; then
  echo "NANOCHAT_DIR must point at a local karpathy/nanochat checkout" >&2
  exit 2
fi

TRIALS="${TRIALS:-300}"
SAMPLER="${SAMPLER:-tpe}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-86400}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/examples/autotune/nanochat_benchmark}"
STUDY_NAME="${STUDY_NAME:-nanochat_benchmark_autotune}"
STORAGE="${STORAGE:-sqlite:///$WORK_DIR/study.db}"
export NANOCHAT_BENCHMARK_TARGET_SECONDS="${NANOCHAT_BENCHMARK_TARGET_SECONDS:-300}"
export NANOCHAT_BENCHMARK_TOKENS_PER_SECOND="${NANOCHAT_BENCHMARK_TOKENS_PER_SECOND:-1666667}"
TRIAL_TIMEOUT_SECONDS="${NANOCHAT_TRIAL_TIMEOUT_SECONDS:-$((NANOCHAT_BENCHMARK_TARGET_SECONDS + 600))}"

PATH="$ROOT_DIR/.venv/bin:$PATH" node "$ROOT_DIR/dist/cli.js" run "$ROOT_DIR/examples/nanochat_benchmark.py" \
  --config "$ROOT_DIR/examples/nanochat_search_space.yaml" \
  --trials "$TRIALS" \
  --sampler "$SAMPLER" \
  --pruner none \
  --n-jobs 1 \
  --timeout-seconds "$TRIAL_TIMEOUT_SECONDS" \
  --time-budget-seconds "$TIME_BUDGET_SECONDS" \
  --storage "$STORAGE" \
  --study-name "$STUDY_NAME" \
  --work-dir "$WORK_DIR" \
  --yes \
  --json
