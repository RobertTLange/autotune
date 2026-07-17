#!/usr/bin/env bash
#SBATCH --job-name=autotune-nanobench
#SBATCH --partition=aisci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=autotune-nanobench-%j.out
#SBATCH --error=autotune-nanobench-%j.err

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/package.json" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  fi
fi
EXAMPLE_DIR="$ROOT_DIR/examples/nanochat"
SCRIPT="$EXAMPLE_DIR/nanochat_benchmark.py"
CONFIG="$EXAMPLE_DIR/nanochat_search_space.yaml"

if [[ -z "${AUTORESEARCH_DIR:-}" ]]; then
  echo "AUTORESEARCH_DIR must point at the pinned karpathy/autoresearch checkout" >&2
  exit 2
fi
if [[ ! -f "$AUTORESEARCH_DIR/train.py" || ! -f "$AUTORESEARCH_DIR/prepare.py" ]]; then
  echo "AUTORESEARCH_DIR does not look like autoresearch: $AUTORESEARCH_DIR" >&2
  echo "Expected train.py and prepare.py at the checkout root." >&2
  exit 2
fi

AGENT="${AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
SAMPLER="${SAMPLER:-tpe}"
PRUNER="${PRUNER:-none}"
N_JOBS="${N_JOBS:-1}"
SAMPLER_SEED="${SAMPLER_SEED:-42}"
if [[ "$N_JOBS" != "1" ]]; then
  echo "N_JOBS must be 1 for the single-GPU comparable protocol" >&2
  exit 2
fi

BASELINE_TRIALS="${BASELINE_TRIALS:-100}"
REFINE_INITIAL_TRIALS="${REFINE_INITIAL_TRIALS:-50}"
REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
REFINE_TRIALS="${REFINE_TRIALS:-25}"
REFINED_TOTAL_TRIALS=$((REFINE_INITIAL_TRIALS + REFINE_ROUNDS * REFINE_TRIALS))
FINALISTS="${FINALISTS:-3}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
VALIDATION_MAX_ATTEMPTS="${VALIDATION_MAX_ATTEMPTS:-2}"

TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-86400}"
if [[ "${NANOCHAT_BENCHMARK_SEED:-42}" != "42" ]]; then
  echo "NANOCHAT_BENCHMARK_SEED must be 42 for discovery" >&2
  exit 2
fi
export NANOCHAT_BENCHMARK_SEED=42
TRIAL_TIMEOUT_SECONDS="${NANOCHAT_TRIAL_TIMEOUT_SECONDS:-1200}"

RUN_GROUP="${RUN_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$EXAMPLE_DIR/autotune/nanobench_ablation/$RUN_GROUP}"
CACHE_ROOT="${AUTOTUNE_RUNTIME_CACHE_DIR:-$OUT_ROOT/runtime_cache}"

if [[ "${ALLOW_UNEQUAL_TRIALS:-0}" != "1" && "$BASELINE_TRIALS" -ne "$REFINED_TOTAL_TRIALS" ]]; then
  echo "Expected equal trial budgets; baseline=$BASELINE_TRIALS refined=$REFINED_TOTAL_TRIALS" >&2
  echo "Set ALLOW_UNEQUAL_TRIALS=1 to run unequal budgets." >&2
  exit 2
fi
AUTORESEARCH_CACHE_DIR="$HOME/.cache/autoresearch"
if [[ ! -f "$AUTORESEARCH_CACHE_DIR/data/shard_06542.parquet" ]]; then
  echo "Missing canonical validation shard under $AUTORESEARCH_CACHE_DIR/data" >&2
  echo "Prepare with: cd \"$AUTORESEARCH_DIR\" && uv sync --frozen && uv run --frozen prepare.py" >&2
  exit 2
fi
if [[ ! -f "$AUTORESEARCH_CACHE_DIR/tokenizer/tokenizer.pkl" || ! -f "$AUTORESEARCH_CACHE_DIR/tokenizer/token_bytes.pt" ]]; then
  echo "Missing autoresearch tokenizer artifacts under $AUTORESEARCH_CACHE_DIR" >&2
  echo "Expected tokenizer/tokenizer.pkl and tokenizer/token_bytes.pt." >&2
  echo "Prepare with: cd \"$AUTORESEARCH_DIR\" && uv sync --frozen && uv run --frozen prepare.py" >&2
  exit 2
fi
NANOCHAT_DATA_IDENTITY_SHA256="$(python3 "$EXAMPLE_DIR/prepare_nanochat_cache.py" verify)"
export NANOCHAT_DATA_IDENTITY_SHA256

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
    --sampler-seed "$SAMPLER_SEED"
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
  NANOCHAT_BENCHMARK_RESULTS_DIR="$work_dir/trial_results" PATH="$ROOT_DIR/.venv/bin:$PATH" "${cmd[@]}" "$@"
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

validation_results=(--result "optuna_baseline=$OUT_ROOT/01_optuna_baseline/results.json")
for ((round = 0; round <= REFINE_ROUNDS; round++)); do
  validation_results+=(
    --result "resets_no_transfer=$OUT_ROOT/02_autotune_resets_no_trial_transfer/results.round_$round.json"
    --result "resets_trial_transfer=$OUT_ROOT/03_autotune_resets_trial_transfer/results.round_$round.json"
  )
done

python3 "$EXAMPLE_DIR/validate_nanochat.py" \
  "${validation_results[@]}" \
  --output-dir "$OUT_ROOT/finalist_validation" \
  --finalists "$FINALISTS" \
  --seeds "$VALIDATION_SEEDS" \
  --timeout-seconds "$TRIAL_TIMEOUT_SECONDS" \
  --max-attempts "$VALIDATION_MAX_ATTEMPTS"

echo
echo "Nanobench ablations complete."
echo "Results root: $OUT_ROOT"
echo "Finalist summary: $OUT_ROOT/finalist_validation/summary.json"
