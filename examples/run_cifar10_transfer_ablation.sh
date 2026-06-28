#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/examples/cifar10_resnet.py"

AGENT="${AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"

SINGLE_SHOT_TRIALS=40
REFINE_INITIAL_TRIALS=20
REFINE_ROUNDS=2
REFINE_TRIALS=10
REFINED_TOTAL_NEW_TRIALS=$((REFINE_INITIAL_TRIALS + REFINE_ROUNDS * REFINE_TRIALS))

RUN_GROUP="${RUN_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/examples/autotune/cifar10_resnet_transfer_ablation/$RUN_GROUP}"
INITIAL_CONFIG="${INITIAL_CONFIG:-$OUT_ROOT/search_space.initial.yaml}"

if [[ "$SINGLE_SHOT_TRIALS" -ne 40 || "$REFINED_TOTAL_NEW_TRIALS" -ne 40 ]]; then
  echo "Expected 40 total new trials, got single=$SINGLE_SHOT_TRIALS refined=$REFINED_TOTAL_NEW_TRIALS" >&2
  exit 1
fi

cd "$ROOT_DIR"
npm run build
mkdir -p "$OUT_ROOT"

analyze_cmd=(
  node "$ROOT_DIR/dist/cli.js" analyze "$SCRIPT"
  --agent "$AGENT"
  --output "$INITIAL_CONFIG"
  --work-dir "$OUT_ROOT/analyze"
)

if [[ -n "$MODEL" ]]; then
  analyze_cmd+=(--model "$MODEL")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  analyze_cmd+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ -f "$INITIAL_CONFIG" ]]; then
  echo "Using existing initial search space: $INITIAL_CONFIG"
else
  echo "Creating shared initial search space: $INITIAL_CONFIG"
  PATH="$ROOT_DIR/.venv/bin:$PATH" "${analyze_cmd[@]}"
fi

run_single_shot_variant() {
  local name="$1"
  local work_dir="$OUT_ROOT/$name"
  local -a cmd=(
    node "$ROOT_DIR/dist/cli.js" run "$SCRIPT"
    --trials "$SINGLE_SHOT_TRIALS"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --agent "$AGENT"
    --work-dir "$work_dir"
    --config "$INITIAL_CONFIG"
    --yes
    --json
  )

  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  if [[ -n "$REASONING_EFFORT" ]]; then
    cmd+=(--reasoning-effort "$REASONING_EFFORT")
  fi

  echo
  echo "==> $name"
  echo "    work_dir: $work_dir"
  echo "    new trial budget: $SINGLE_SHOT_TRIALS"
  PATH="$ROOT_DIR/.venv/bin:$PATH" "${cmd[@]}"
}

run_refined_variant() {
  local name="$1"
  shift
  local work_dir="$OUT_ROOT/$name"
  local -a cmd=(
    node "$ROOT_DIR/dist/cli.js" run "$SCRIPT"
    --trials "$REFINE_INITIAL_TRIALS"
    --refine-rounds "$REFINE_ROUNDS"
    --refine-trials "$REFINE_TRIALS"
    --refine-mode auto
    --timeout-seconds "$TIMEOUT_SECONDS"
    --agent "$AGENT"
    --work-dir "$work_dir"
    --config "$INITIAL_CONFIG"
    --yes
    --json
  )

  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  if [[ -n "$REASONING_EFFORT" ]]; then
    cmd+=(--reasoning-effort "$REASONING_EFFORT")
  fi

  echo
  echo "==> $name"
  echo "    work_dir: $work_dir"
  echo "    new trial budget: $REFINE_INITIAL_TRIALS + $REFINE_ROUNDS x $REFINE_TRIALS = $REFINED_TOTAL_NEW_TRIALS"
  PATH="$ROOT_DIR/.venv/bin:$PATH" "${cmd[@]}" "$@"
}

run_single_shot_variant "01_single_shot"

run_refined_variant "02_no_transfer" \
  --no-refine-transfer-fixed-params \
  --no-refine-transfer-trials

run_refined_variant "03_full_transfer"

run_refined_variant "04_fixed_params_only" \
  --no-refine-transfer-trials

run_refined_variant "05_trial_seeding_only" \
  --no-refine-transfer-fixed-params

echo
echo "All CIFAR-10 transfer ablation runs completed."
echo "Results root: $OUT_ROOT"
