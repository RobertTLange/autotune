#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
SECONDS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$SCRIPT_DIR/benchmark.py"
PROCESS_SUPERVISOR="$SCRIPT_DIR/supervise_process.py"
AUTOTUNE_CLI="${AUTOTUNE_CLI:-$ROOT_DIR/dist/cli.js}"

OBJECTIVES=(sphere ellipsoid rosenbrock rastrigin)
ACTIVE_PIDS=()
ACTIVE_NAMES=()
COMMAND=()

TOTAL_TRIALS="${TOTAL_TRIALS:-100}"
REFINE_INITIAL_TRIALS="${REFINE_INITIAL_TRIALS:-50}"
REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
REFINE_TRIALS="${REFINE_TRIALS:-25}"
AGENT="${AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
CENTAUR_LLM_PROBABILITY="${CENTAUR_LLM_PROBABILITY:-0.3}"
CENTAUR_WARMUP_TRIALS="${CENTAUR_WARMUP_TRIALS:-10}"
SEED_COUNT="${SEED_COUNT:-10}"
BUILD="${BUILD:-auto}"
TERMINATION_GRACE_SECONDS="${TERMINATION_GRACE_SECONDS:-10}"
RUN_GROUP="${RUN_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/autotune/experiments/$RUN_GROUP}"

fail() {
  echo "error: $*" >&2
  exit 2
}

validate_integer() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  local maximum="$4"
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || fail "$name must be a canonical non-negative integer"
  local number=$((10#$value))
  ((number >= minimum && number <= maximum)) || fail "$name must be between $minimum and $maximum"
}

validate_configuration() {
  validate_integer TOTAL_TRIALS "$TOTAL_TRIALS" 1 1000000
  validate_integer REFINE_INITIAL_TRIALS "$REFINE_INITIAL_TRIALS" 1 1000000
  validate_integer REFINE_ROUNDS "$REFINE_ROUNDS" 1 100
  validate_integer REFINE_TRIALS "$REFINE_TRIALS" 1 1000000
  validate_integer CENTAUR_WARMUP_TRIALS "$CENTAUR_WARMUP_TRIALS" 0 1000000
  validate_integer SEED_COUNT "$SEED_COUNT" 1 100
  validate_integer TERMINATION_GRACE_SECONDS "$TERMINATION_GRACE_SECONDS" 1 60
  [[ "$CENTAUR_LLM_PROBABILITY" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] ||
    fail "CENTAUR_LLM_PROBABILITY must be between 0 and 1"
  [[ "$RUN_GROUP" =~ ^[A-Za-z0-9._-]+$ ]] || fail "RUN_GROUP contains unsupported characters"
  [[ -n "$AGENT" ]] || fail "AGENT must not be empty"
  [[ -n "$OUT_ROOT" && "$OUT_ROOT" != "/" ]] || fail "OUT_ROOT must be a non-root path"
  [[ "$BUILD" == "auto" || "$BUILD" == "0" || "$BUILD" == "1" ]] || fail "BUILD must be auto, 0, or 1"
  if [[ -n "$REASONING_EFFORT" ]]; then
    [[ "$REASONING_EFFORT" =~ ^(low|medium|high|xhigh)$ ]] ||
      fail "REASONING_EFFORT must be low, medium, high, or xhigh"
  fi

  local refined_total=$((10#$REFINE_INITIAL_TRIALS + 10#$REFINE_ROUNDS * 10#$REFINE_TRIALS))
  ((refined_total == 10#$TOTAL_TRIALS)) ||
    fail "equal budgets required: TOTAL_TRIALS=$TOTAL_TRIALS, refinement total=$refined_total"
}

prepare_cli() {
  [[ -f "$SCRIPT" ]] || fail "missing benchmark: $SCRIPT"
  [[ -f "$PROCESS_SUPERVISOR" ]] || fail "missing process supervisor: $PROCESS_SUPERVISOR"
  for objective in "${OBJECTIVES[@]}"; do
    [[ -f "$SCRIPT_DIR/${objective}_search_space.yaml" ]] || fail "missing search space for $objective"
  done

  if [[ "$BUILD" == "1" || ( "$BUILD" == "auto" && -f "$ROOT_DIR/src/cli.ts" ) ]]; then
    (cd "$ROOT_DIR" && npm run build)
  fi
  [[ -f "$AUTOTUNE_CLI" ]] || fail "missing Autotune CLI: $AUTOTUNE_CLI"
}

make_command() {
  local objective="$1"
  local variant="$2"
  local trials="$3"
  local sampler="$4"
  local seed="$5"
  local work_dir="$OUT_ROOT/$objective/$variant/seed_$seed"
  local study_name="bbob_${objective}_${variant}_seed_${seed}"
  local guidance="Keep function fixed to $objective. Refine only x1-x5 bounds; preserve minimization and metric semantics."

  COMMAND=(
    node "$AUTOTUNE_CLI" run "$SCRIPT"
    --config "$SCRIPT_DIR/${objective}_search_space.yaml"
    --trials "$trials"
    --sampler "$sampler"
    --pruner none
    --direction minimize
    --n-jobs 1
    --storage "sqlite:///$work_dir/study.db"
    --study-name "$study_name"
    --work-dir "$work_dir"
    --agent "$AGENT"
    --agent-guidance "$guidance"
    --yes
    --json
  )

  [[ -z "$MODEL" ]] || COMMAND+=(--model "$MODEL")
  [[ -z "$REASONING_EFFORT" ]] || COMMAND+=(--reasoning-effort "$REASONING_EFFORT")
  [[ "$sampler" == "centaur" ]] || COMMAND+=(--sampler-seed "$seed")

  case "$variant" in
    01_base_cmaes)
      COMMAND+=(--refine-rounds 0)
      ;;
    02_reset_no_transfer)
      COMMAND+=(
        --refine-rounds "$REFINE_ROUNDS"
        --refine-trials "$REFINE_TRIALS"
        --refine-mode auto
        --no-refine-transfer-fixed-params
        --no-refine-transfer-trials
      )
      ;;
    03_reset_with_transfer)
      COMMAND+=(
        --refine-rounds "$REFINE_ROUNDS"
        --refine-trials "$REFINE_TRIALS"
        --refine-mode auto
        --no-refine-transfer-fixed-params
      )
      ;;
    04_centaur)
      COMMAND+=(
        --refine-rounds 0
        --centaur-llm-probability "$CENTAUR_LLM_PROBABILITY"
        --centaur-warmup-trials "$CENTAUR_WARMUP_TRIALS"
        --centaur-seed "$seed"
      )
      ;;
    *)
      fail "unknown experiment variant: $variant"
      ;;
  esac
}

terminate_active() {
  local pid
  ((${#ACTIVE_PIDS[@]} > 0)) || return 0
  local -a pids=("${ACTIVE_PIDS[@]}")

  for pid in "${pids[@]}"; do
    [[ "$pid" == "0" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    [[ "$pid" == "0" ]] || wait "$pid" 2>/dev/null || true
  done
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  terminate_active
  exit "$status"
}

run_phase() {
  local seed="$1"
  local variant="$2"
  local trials="$3"
  local sampler="$4"
  local total_trials="$5"
  local objective
  local work_dir
  local index
  local launched_pid
  local failures=0

  ACTIVE_PIDS=()
  ACTIVE_NAMES=()
  echo
  echo "==> seed $((seed + 1))/$SEED_COUNT: $variant ($sampler, $total_trials new trials per objective)"
  if [[ "$trials" != "$total_trials" ]]; then
    echo "    budget: $trials + $REFINE_ROUNDS x $REFINE_TRIALS = $total_trials"
  fi
  for objective in "${OBJECTIVES[@]}"; do
    work_dir="$OUT_ROOT/$objective/$variant/seed_$seed"
    mkdir -p -- "$work_dir"
    make_command "$objective" "$variant" "$trials" "$sampler" "$seed"
    PATH="$ROOT_DIR/.venv/bin:$PATH" python3 "$PROCESS_SUPERVISOR" \
      --grace-seconds "$TERMINATION_GRACE_SECONDS" -- \
      "${COMMAND[@]}" >"$work_dir/run.log" 2>&1 &
    launched_pid=$!
    ACTIVE_PIDS+=("$launched_pid")
    ACTIVE_NAMES+=("$objective/$variant/seed_$seed")
    echo "    started $objective (pid $launched_pid, log $work_dir/run.log)"
  done

  for ((index = 0; index < ${#ACTIVE_PIDS[@]}; index += 1)); do
    if [[ "${ACTIVE_PIDS[$index]}" == "0" ]]; then
      echo "    completed ${ACTIVE_NAMES[$index]}"
      continue
    fi
    if wait "${ACTIVE_PIDS[$index]}"; then
      echo "    completed ${ACTIVE_NAMES[$index]}"
    else
      local status=$?
      echo "    failed ${ACTIVE_NAMES[$index]} (exit $status)" >&2
      failures=$((failures + 1))
    fi
    ACTIVE_PIDS[$index]=0
  done
  ACTIVE_PIDS=()
  ((failures == 0))
}

main() {
  local seed
  validate_configuration
  prepare_cli
  mkdir -p -- "$(dirname "$OUT_ROOT")"
  mkdir -m 700 -- "$OUT_ROOT" 2>/dev/null || fail "output root already exists or cannot be created: $OUT_ROOT"

  for ((seed = 0; seed < 10#$SEED_COUNT; seed += 1)); do
    run_phase "$seed" 01_base_cmaes "$TOTAL_TRIALS" cmaes "$TOTAL_TRIALS"
    run_phase "$seed" 02_reset_no_transfer "$REFINE_INITIAL_TRIALS" cmaes "$TOTAL_TRIALS"
    run_phase "$seed" 03_reset_with_transfer "$REFINE_INITIAL_TRIALS" cmaes "$TOTAL_TRIALS"
    run_phase "$seed" 04_centaur "$TOTAL_TRIALS" centaur "$TOTAL_TRIALS"
  done

  echo
  echo "BBOB experiments complete: $OUT_ROOT"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
main "$@"
