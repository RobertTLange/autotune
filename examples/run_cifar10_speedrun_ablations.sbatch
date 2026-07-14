#!/usr/bin/env bash
#SBATCH --job-name=autotune-cifar10-speedrun
#SBATCH --partition=aisci,a3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/package.json" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

if [[ ! -f "$ROOT_DIR/package.json" || ! -d "$ROOT_DIR/examples" ]]; then
  echo "Could not locate autotune repo root at ROOT_DIR=$ROOT_DIR" >&2
  echo "Submit from the repo root or set ROOT_DIR=/path/to/autotune." >&2
  exit 1
fi

SCRIPT="$ROOT_DIR/examples/cifar10_speedrun.py"

AGENT="${AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
AGENT_GUIDANCE="${AGENT_GUIDANCE:-Tune only training hyperparameters and preserve the CIFAR-10 speedrun scoring protocol. Do not tune CIFAR10_SPEEDRUN_* environment controls or data/cache paths.}"

export CIFAR10_SPEEDRUN_NUM_RUNS="${CIFAR10_SPEEDRUN_NUM_RUNS:-100}"
export CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN="${CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN:-3.0}"
export CIFAR10_SPEEDRUN_DATA_DIR="${CIFAR10_SPEEDRUN_DATA_DIR:-$HOME/data/cifar10}"

SINGLE_SHOT_TRIALS="${SINGLE_SHOT_TRIALS:-100}"
REFINE_INITIAL_TRIALS="${REFINE_INITIAL_TRIALS:-50}"
REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
REFINE_TRIALS="${REFINE_TRIALS:-25}"
REFINED_TOTAL_NEW_TRIALS=$((REFINE_INITIAL_TRIALS + REFINE_ROUNDS * REFINE_TRIALS))

RUN_GROUP="${RUN_GROUP:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/examples/autotune/cifar10_speedrun_ablations/$RUN_GROUP}"
INITIAL_CONFIG="${INITIAL_CONFIG:-$OUT_ROOT/search_space.initial.yaml}"
export CIFAR10_SPEEDRUN_RESULTS_DIR="${CIFAR10_SPEEDRUN_RESULTS_DIR:-$OUT_ROOT/trial_metrics}"

if [[ "$SINGLE_SHOT_TRIALS" -ne "$REFINED_TOTAL_NEW_TRIALS" ]]; then
  echo "Expected equal new-trial budgets, got baseline=$SINGLE_SHOT_TRIALS refined=$REFINED_TOTAL_NEW_TRIALS" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "nvidia-smi failed; continuing so the workload can report the real CUDA error if needed." >&2
fi

if [[ -n "${LOAD_MODULES:-}" ]]; then
  # shellcheck disable=SC2086
  module load $LOAD_MODULES
fi

if [[ -n "${SETUP_COMMAND:-}" ]]; then
  eval "$SETUP_COMMAND"
fi

cd "$ROOT_DIR"
npm run build
mkdir -p "$OUT_ROOT"

common_agent_flags=()
if [[ -n "$MODEL" ]]; then
  common_agent_flags+=(--model "$MODEL")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  common_agent_flags+=(--reasoning-effort "$REASONING_EFFORT")
fi

analyze_cmd=(
  node "$ROOT_DIR/dist/cli.js" analyze "$SCRIPT"
  --agent "$AGENT"
  --agent-guidance "$AGENT_GUIDANCE"
  --output "$INITIAL_CONFIG"
  --work-dir "$OUT_ROOT/analyze"
  "${common_agent_flags[@]}"
)

if [[ -f "$INITIAL_CONFIG" ]]; then
  echo "Using existing initial search space: $INITIAL_CONFIG"
else
  echo "Creating shared initial search space: $INITIAL_CONFIG"
  PATH="$ROOT_DIR/.venv/bin:$PATH" "${analyze_cmd[@]}"
fi

run_baseline() {
  local name="$1"
  local work_dir="$OUT_ROOT/$name"
  local -a cmd=(
    node "$ROOT_DIR/dist/cli.js" run "$SCRIPT"
    --trials "$SINGLE_SHOT_TRIALS"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --direction maximize
    --agent "$AGENT"
    --agent-guidance "$AGENT_GUIDANCE"
    --work-dir "$work_dir"
    --config "$INITIAL_CONFIG"
    --yes
    --json
    "${common_agent_flags[@]}"
  )

  echo
  echo "==> $name"
  echo "    description: base Optuna speedrun search baseline"
  echo "    work_dir: $work_dir"
  echo "    new trial budget: $SINGLE_SHOT_TRIALS"
  echo "    speedrun timed runs per trial: $CIFAR10_SPEEDRUN_NUM_RUNS"
  PATH="$ROOT_DIR/.venv/bin:$PATH" "${cmd[@]}"
}

run_reset_variant() {
  local name="$1"
  local description="$2"
  shift 2
  local work_dir="$OUT_ROOT/$name"
  local -a cmd=(
    node "$ROOT_DIR/dist/cli.js" run "$SCRIPT"
    --trials "$REFINE_INITIAL_TRIALS"
    --refine-rounds "$REFINE_ROUNDS"
    --refine-trials "$REFINE_TRIALS"
    --refine-mode auto
    --timeout-seconds "$TIMEOUT_SECONDS"
    --direction maximize
    --agent "$AGENT"
    --agent-guidance "$AGENT_GUIDANCE"
    --work-dir "$work_dir"
    --config "$INITIAL_CONFIG"
    --yes
    --json
    "${common_agent_flags[@]}"
  )

  echo
  echo "==> $name"
  echo "    description: $description"
  echo "    work_dir: $work_dir"
  echo "    new trial budget: $REFINE_INITIAL_TRIALS + $REFINE_ROUNDS x $REFINE_TRIALS = $REFINED_TOTAL_NEW_TRIALS"
  echo "    speedrun timed runs per trial: $CIFAR10_SPEEDRUN_NUM_RUNS"
  PATH="$ROOT_DIR/.venv/bin:$PATH" "${cmd[@]}" "$@"
}

run_baseline "01_base_optuna"

run_reset_variant "02_resets_no_trial_transfer" \
  "Autotune speedrun search-space resets with no fixed-parameter or trial transfer" \
  --no-refine-transfer-fixed-params \
  --no-refine-transfer-trials

run_reset_variant "03_resets_trial_transfer" \
  "Autotune speedrun search-space resets with trial transfer only" \
  --no-refine-transfer-fixed-params

echo
echo "All requested CIFAR-10 speedrun ablations completed."
echo "Results root: $OUT_ROOT"
