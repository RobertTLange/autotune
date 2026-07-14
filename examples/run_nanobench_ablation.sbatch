#!/usr/bin/env bash
#SBATCH --job-name=autotune-nanobench
#SBATCH --partition=aisci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=autotune-nanobench-%j.out
#SBATCH --error=autotune-nanobench-%j.err

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/package.json" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi
SCRIPT="$ROOT_DIR/examples/nanochat_benchmark.py"
CONFIG="$ROOT_DIR/examples/nanochat_search_space.yaml"

if [[ -z "${NANOCHAT_DIR:-}" ]]; then
  echo "NANOCHAT_DIR must point at a local karpathy/nanochat checkout" >&2
  exit 2
fi
if [[ ! -f "$NANOCHAT_DIR/scripts/base_train.py" ]]; then
  echo "NANOCHAT_DIR does not look like nanochat: $NANOCHAT_DIR" >&2
  echo "Expected file: $NANOCHAT_DIR/scripts/base_train.py" >&2
  exit 2
fi

AGENT="${AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
SAMPLER="${SAMPLER:-tpe}"
PRUNER="${PRUNER:-none}"
N_JOBS="${N_JOBS:-1}"

BASELINE_TRIALS="${BASELINE_TRIALS:-100}"
REFINE_INITIAL_TRIALS="${REFINE_INITIAL_TRIALS:-50}"
REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
REFINE_TRIALS="${REFINE_TRIALS:-25}"
REFINED_TOTAL_TRIALS=$((REFINE_INITIAL_TRIALS + REFINE_ROUNDS * REFINE_TRIALS))

TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-86400}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_BENCHMARK_NPROC_PER_NODE="${NANOCHAT_BENCHMARK_NPROC_PER_NODE:-8}"
export NANOCHAT_BENCHMARK_TARGET_SECONDS="${NANOCHAT_BENCHMARK_TARGET_SECONDS:-300}"
export NANOCHAT_BENCHMARK_TOKENS_PER_SECOND="${NANOCHAT_BENCHMARK_TOKENS_PER_SECOND:-10000000}"
TRIAL_TIMEOUT_SECONDS="${NANOCHAT_TRIAL_TIMEOUT_SECONDS:-$((NANOCHAT_BENCHMARK_TARGET_SECONDS + 600))}"

RUN_GROUP="${RUN_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/examples/autotune/nanobench_ablation/$RUN_GROUP}"
CACHE_ROOT="${AUTOTUNE_RUNTIME_CACHE_DIR:-$OUT_ROOT/runtime_cache}"

if [[ "${ALLOW_UNEQUAL_TRIALS:-0}" != "1" && "$BASELINE_TRIALS" -ne "$REFINED_TOTAL_TRIALS" ]]; then
  echo "Expected equal trial budgets; baseline=$BASELINE_TRIALS refined=$REFINED_TOTAL_TRIALS" >&2
  echo "Set ALLOW_UNEQUAL_TRIALS=1 to run unequal budgets." >&2
  exit 2
fi
if ! compgen -G "$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet" >/dev/null; then
  echo "Missing nanochat parquet data under $NANOCHAT_BASE_DIR/base_data_climbmix" >&2
  echo "Prepare with: cd \"$NANOCHAT_DIR\" && NANOCHAT_BASE_DIR=\"$NANOCHAT_BASE_DIR\" .venv/bin/python -m nanochat.dataset -n 8" >&2
  exit 2
fi
if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" || ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]]; then
  echo "Missing nanochat tokenizer artifacts under NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR" >&2
  echo "Expected tokenizer/tokenizer.pkl and tokenizer/token_bytes.pt." >&2
  echo "Prepare after dataset download with: cd \"$NANOCHAT_DIR\" && NANOCHAT_BASE_DIR=\"$NANOCHAT_BASE_DIR\" .venv/bin/python -m scripts.tok_train --max-chars=2000000000" >&2
  exit 2
fi

cd "$ROOT_DIR"
npm run build
mkdir -p "$OUT_ROOT" "$CACHE_ROOT/tmp" "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/triton"

export TMPDIR="$CACHE_ROOT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"

base_cmd() {
  local name="$1"
  local work_dir="$OUT_ROOT/$name"
  local study_name="nanobench_${RUN_GROUP}_${name}"
  local storage="sqlite:///$work_dir/study.db"

  mkdir -p "$work_dir"
  echo
  echo "==> $name"
  echo "    work_dir: $work_dir"
  echo "    storage: $storage"
  echo "    study_name: $study_name"
}

run_variant() {
  local name="$1"
  shift
  local trials="$1"
  shift
  local work_dir="$OUT_ROOT/$name"
  local study_name="nanobench_${RUN_GROUP}_${name}"
  local storage="sqlite:///$work_dir/study.db"
  local -a cmd=(
    node "$ROOT_DIR/dist/cli.js" run "$SCRIPT"
    --config "$CONFIG"
    --trials "$trials"
    --sampler "$SAMPLER"
    --pruner "$PRUNER"
    --n-jobs "$N_JOBS"
    --timeout-seconds "$TRIAL_TIMEOUT_SECONDS"
    --time-budget-seconds "$TIME_BUDGET_SECONDS"
    --storage "$storage"
    --study-name "$study_name"
    --work-dir "$work_dir"
    --agent "$AGENT"
    --yes
    --json
  )

  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  if [[ -n "$REASONING_EFFORT" ]]; then
    cmd+=(--reasoning-effort "$REASONING_EFFORT")
  fi

  base_cmd "$name"
  PATH="$ROOT_DIR/.venv/bin:$PATH" "${cmd[@]}" "$@"
}

run_variant "01_optuna_baseline" "$BASELINE_TRIALS"

run_variant "02_autotune_resets_no_trial_transfer" "$REFINE_INITIAL_TRIALS" \
  --refine-rounds "$REFINE_ROUNDS" \
  --refine-trials "$REFINE_TRIALS" \
  --refine-mode auto \
  --no-refine-transfer-fixed-params \
  --no-refine-transfer-trials

run_variant "03_autotune_resets_trial_transfer" "$REFINE_INITIAL_TRIALS" \
  --refine-rounds "$REFINE_ROUNDS" \
  --refine-trials "$REFINE_TRIALS" \
  --refine-mode auto \
  --no-refine-transfer-fixed-params

echo
echo "Nanobench ablations complete."
echo "Results root: $OUT_ROOT"
