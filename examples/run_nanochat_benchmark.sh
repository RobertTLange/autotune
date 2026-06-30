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

PATH="$ROOT_DIR/.venv/bin:$PATH" node "$ROOT_DIR/dist/cli.js" run "$ROOT_DIR/examples/nanochat_benchmark.py" \
  --config "$ROOT_DIR/examples/nanochat_search_space.yaml" \
  --trials "$TRIALS" \
  --sampler "$SAMPLER" \
  --pruner none \
  --n-jobs 1 \
  --timeout-seconds "${NANOCHAT_TRIAL_TIMEOUT_SECONDS:-1200}" \
  --time-budget-seconds "$TIME_BUDGET_SECONDS" \
  --work-dir "$WORK_DIR" \
  --yes \
  --json
