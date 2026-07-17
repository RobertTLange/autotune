#!/usr/bin/env bash
#SBATCH --job-name=autotune-nanobench
#SBATCH --partition=aisci
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
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
if [[ "$SAMPLER" == "centaur" ]]; then
  echo "SAMPLER must be a non-Centaur Optuna sampler; Centaur has a dedicated fourth arm" >&2
  exit 2
fi

BASELINE_TRIALS="${BASELINE_TRIALS:-100}"
REFINE_INITIAL_TRIALS="${REFINE_INITIAL_TRIALS:-50}"
REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
REFINE_TRIALS="${REFINE_TRIALS:-25}"
REFINED_TOTAL_TRIALS=$((REFINE_INITIAL_TRIALS + REFINE_ROUNDS * REFINE_TRIALS))
CENTAUR_TRIALS="${CENTAUR_TRIALS:-$BASELINE_TRIALS}"
CENTAUR_LLM_PROBABILITY="${CENTAUR_LLM_PROBABILITY:-0.3}"
CENTAUR_WARMUP_TRIALS="${CENTAUR_WARMUP_TRIALS:-10}"
CENTAUR_SEED="${CENTAUR_SEED:-42}"
FINALISTS="${FINALISTS:-3}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
VALIDATION_MAX_ATTEMPTS="${VALIDATION_MAX_ATTEMPTS:-2}"
VALIDATION_ONLY="${VALIDATION_ONLY:-0}"
CANCEL_GRACE_SECONDS="${CANCEL_GRACE_SECONDS:-5}"
if [[ "$VALIDATION_ONLY" != "0" && "$VALIDATION_ONLY" != "1" ]]; then
  echo "VALIDATION_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$CANCEL_GRACE_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "CANCEL_GRACE_SECONDS must be a non-negative integer" >&2
  exit 2
fi

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

if [[ "${ALLOW_UNEQUAL_TRIALS:-0}" != "1" && \
  ("$BASELINE_TRIALS" -ne "$REFINED_TOTAL_TRIALS" || "$BASELINE_TRIALS" -ne "$CENTAUR_TRIALS") ]]; then
  echo "Expected equal trial budgets; baseline=$BASELINE_TRIALS refined=$REFINED_TOTAL_TRIALS centaur=$CENTAUR_TRIALS" >&2
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

DISCOVERY_MANIFEST="$OUT_ROOT/discovery_complete.json"
if [[ "$VALIDATION_ONLY" == "1" ]]; then
  if [[ ! -d "$OUT_ROOT" ]]; then
    echo "Validation run root does not exist: $OUT_ROOT" >&2
    exit 2
  fi
  REFINE_ROUNDS="$(python3 "$EXAMPLE_DIR/manage_nanochat_run.py" verify --manifest "$DISCOVERY_MANIFEST")"
else
  mkdir -p "$(dirname "$OUT_ROOT")"
  if ! mkdir "$OUT_ROOT"; then
    echo "Run root already exists; choose a new RUN_GROUP or OUT_ROOT: $OUT_ROOT" >&2
    exit 2
  fi
fi
mkdir -p "$CACHE_ROOT"

cd "$ROOT_DIR"
npm run build

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
  echo "    log: $work_dir/run.log"
}

run_variant() {
  local name="$1"
  shift
  local trials="$1"
  shift
  local sampler="$1"
  shift
  local work_dir="$OUT_ROOT/$name"
  local cache_dir="$CACHE_ROOT/$name"
  local study_name="nanobench_${RUN_GROUP}_${name}"
  local storage="sqlite:///$work_dir/study.db"
  local -a cmd=(
    node "$ROOT_DIR/dist/cli.js" run "$SCRIPT"
    --config "$CONFIG"
    --trials "$trials"
    --sampler "$sampler"
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
  if [[ "$sampler" != "centaur" ]]; then
    cmd+=(--sampler-seed "$SAMPLER_SEED")
  fi

  mkdir -p "$cache_dir/tmp" "$cache_dir/torchinductor" "$cache_dir/triton"
  exec srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:1 env \
    TMPDIR="$cache_dir/tmp" \
    TORCHINDUCTOR_CACHE_DIR="$cache_dir/torchinductor" \
    TRITON_CACHE_DIR="$cache_dir/triton" \
    NANOCHAT_BENCHMARK_RESULTS_DIR="$work_dir/trial_results" \
    PATH="$ROOT_DIR/.venv/bin:$PATH" \
    "${cmd[@]}" "$@"
}

active_variant_pids=()
declare -A variant_names_by_pid=()
declare -A variant_logs_by_pid=()
discovery_arms=(
  "01_optuna_baseline"
  "02_autotune_resets_no_trial_transfer"
  "03_autotune_resets_trial_transfer"
  "04_centaur"
)

launch_variant() {
  local name="$1"
  local work_dir="$OUT_ROOT/$name"
  base_cmd "$name"
  run_variant "$@" >"$work_dir/run.log" 2>&1 &
  local pid="$!"
  active_variant_pids+=("$pid")
  variant_names_by_pid["$pid"]="$name"
  variant_logs_by_pid["$pid"]="$work_dir/run.log"
  echo "    started: pid $pid"
}

remove_active_variant() {
  local finished_pid="$1"
  local pid
  local remaining=()
  for pid in "${active_variant_pids[@]}"; do
    [[ "$pid" == "$finished_pid" ]] || remaining+=("$pid")
  done
  active_variant_pids=("${remaining[@]}")
}

cancel_active_variants() {
  local pid running
  local deadline=$((SECONDS + CANCEL_GRACE_SECONDS))
  for pid in "${active_variant_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  while ((SECONDS < deadline)); do
    running=0
    for pid in "${active_variant_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && running=1
    done
    ((running == 0)) && break
    sleep 0.1
  done
  for pid in "${active_variant_pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  for pid in "${active_variant_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  active_variant_pids=()
}

trap cancel_active_variants EXIT
trap 'exit 130' INT TERM

wait_for_variants() {
  local pid status name log
  while ((${#active_variant_pids[@]} > 0)); do
    for pid in "${active_variant_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      if wait "$pid"; then
        status=0
      else
        status="$?"
      fi
      name="${variant_names_by_pid[$pid]}"
      log="${variant_logs_by_pid[$pid]}"
      remove_active_variant "$pid"
      if [[ "$status" -eq 0 ]]; then
        echo "<== completed $name"
        continue
      fi
      echo "<== failed $name (exit $status; log $log)" >&2
      tail -n 40 "$log" >&2 || true
      cancel_active_variants
      return 1
    done
    ((${#active_variant_pids[@]} == 0)) || sleep 0.1
  done
}

if [[ "$VALIDATION_ONLY" == "0" ]]; then
  launch_variant "01_optuna_baseline" "$BASELINE_TRIALS" "$SAMPLER"

  launch_variant "02_autotune_resets_no_trial_transfer" "$REFINE_INITIAL_TRIALS" "$SAMPLER" \
    --refine-rounds "$REFINE_ROUNDS" \
    --refine-trials "$REFINE_TRIALS" \
    --refine-mode auto \
    --no-refine-transfer-fixed-params \
    --no-refine-transfer-trials

  launch_variant "03_autotune_resets_trial_transfer" "$REFINE_INITIAL_TRIALS" "$SAMPLER" \
    --refine-rounds "$REFINE_ROUNDS" \
    --refine-trials "$REFINE_TRIALS" \
    --refine-mode auto \
    --no-refine-transfer-fixed-params

  launch_variant "04_centaur" "$CENTAUR_TRIALS" centaur \
    --refine-rounds 0 \
    --centaur-llm-probability "$CENTAUR_LLM_PROBABILITY" \
    --centaur-warmup-trials "$CENTAUR_WARMUP_TRIALS" \
    --centaur-seed "$CENTAUR_SEED"

  if ! wait_for_variants; then
    exit 1
  fi

  manifest_results=(
    --result "optuna_baseline=$BASELINE_TRIALS:$OUT_ROOT/01_optuna_baseline/results.json"
    --result "centaur=$CENTAUR_TRIALS:$OUT_ROOT/04_centaur/results.json"
  )
  for ((round = 0; round <= REFINE_ROUNDS; round++)); do
    expected_trials="$REFINE_TRIALS"
    [[ "$round" -eq 0 ]] && expected_trials="$REFINE_INITIAL_TRIALS"
    manifest_results+=(
      --result "resets_no_transfer_round_$round=$expected_trials:$OUT_ROOT/02_autotune_resets_no_trial_transfer/results.round_$round.json"
      --result "resets_trial_transfer_round_$round=$expected_trials:$OUT_ROOT/03_autotune_resets_trial_transfer/results.round_$round.json"
    )
  done
  python3 "$EXAMPLE_DIR/manage_nanochat_run.py" create \
    --manifest "$DISCOVERY_MANIFEST" \
    --refine-rounds "$REFINE_ROUNDS" \
    "${manifest_results[@]}"
fi

validation_results=(
  --result "optuna_baseline=$OUT_ROOT/01_optuna_baseline/results.json"
  --result "centaur=$OUT_ROOT/04_centaur/results.json"
)
for ((round = 0; round <= REFINE_ROUNDS; round++)); do
  validation_results+=(
    --result "resets_no_transfer=$OUT_ROOT/02_autotune_resets_no_trial_transfer/results.round_$round.json"
    --result "resets_trial_transfer=$OUT_ROOT/03_autotune_resets_trial_transfer/results.round_$round.json"
  )
done

mkdir -p "$CACHE_ROOT/validation/tmp" "$CACHE_ROOT/validation/torchinductor" "$CACHE_ROOT/validation/triton"
srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:1 env \
  TMPDIR="$CACHE_ROOT/validation/tmp" \
  TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/validation/torchinductor" \
  TRITON_CACHE_DIR="$CACHE_ROOT/validation/triton" \
  PATH="$ROOT_DIR/.venv/bin:$PATH" \
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
