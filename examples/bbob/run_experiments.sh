#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
SECONDS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$SCRIPT_DIR/benchmark.py"
AUTOTUNE_CLI="${AUTOTUNE_CLI:-$ROOT_DIR/dist/cli.js}"

OBJECTIVES=(sphere ellipsoid rosenbrock rastrigin)
ACTIVE_PIDS=()
ACTIVE_NAMES=()
ACTIVE_GROUPS=()
ACTIVE_IDENTITIES=()
ACTIVE_DESCENDANTS=()
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
CENTAUR_SEED="${CENTAUR_SEED:-0}"
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
  validate_integer CENTAUR_SEED "$CENTAUR_SEED" 0 2147483647
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
  process_identity "$$" >/dev/null || fail "process identity tracking is unavailable"
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
  local work_dir="$OUT_ROOT/$objective/$variant"
  local study_name="bbob_${RUN_GROUP}_${objective}_${variant}"
  local guidance="Keep function fixed to $objective. Refine only x1/x2 bounds; preserve minimization and metric semantics."

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
        --centaur-seed "$CENTAUR_SEED"
      )
      ;;
    *)
      fail "unknown experiment variant: $variant"
      ;;
  esac
}

descendants_of() {
  local parent="$1"
  local child
  command -v pgrep >/dev/null 2>&1 || return 0
  while IFS= read -r child; do
    [[ "$child" =~ ^[0-9]+$ ]] || continue
    descendants_of "$child"
    printf '%s\n' "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
}

process_identity() {
  local pid="$1"
  local stat
  local started
  local -a fields
  if [[ -r "/proc/$pid/stat" ]]; then
    IFS= read -r stat < "/proc/$pid/stat" || return 1
    read -r -a fields <<< "${stat##*) }"
    ((${#fields[@]} >= 20)) || return 1
    printf '%s:%s\n' "$pid" "${fields[19]}"
    return 0
  fi

  started="$(ps -o lstart= -p "$pid" 2>/dev/null)" || return 1
  [[ -n "$started" ]] || return 1
  printf '%s:%s\n' "$pid" "$started"
}

identity_matches() {
  local pid="$1"
  local expected="$2"
  local actual
  actual="$(process_identity "$pid")" || return 1
  [[ "$actual" == "$expected" ]]
}

descendant_identities_of() {
  local parent="$1"
  local descendant
  local identity
  while IFS= read -r descendant; do
    [[ -n "$descendant" ]] || continue
    identity="$(process_identity "$descendant")" || continue
    printf '%s\n' "$identity"
  done <<< "$(descendants_of "$parent")"
}

experiment_processes_alive() {
  local index="$1"
  local pid="${ACTIVE_PIDS[$index]}"
  local identity
  local descendant_pid
  if identity_matches "$pid" "${ACTIVE_IDENTITIES[$index]}"; then
    return 0
  fi
  while IFS= read -r identity; do
    [[ -n "$identity" ]] || continue
    descendant_pid="${identity%%:*}"
    if identity_matches "$descendant_pid" "$identity"; then
      return 0
    fi
  done <<< "${ACTIVE_DESCENDANTS[$index]}"
  return 1
}

active_processes_alive() {
  local index
  for ((index = 0; index < ${#ACTIVE_PIDS[@]}; index += 1)); do
    if [[ "${ACTIVE_PIDS[$index]}" != "0" ]] && experiment_processes_alive "$index"; then
      return 0
    fi
  done
  return 1
}

force_kill_after_grace() {
  local index
  local pid
  local group
  local identity
  local descendant_pid
  local deadline=$((SECONDS + 10#$TERMINATION_GRACE_SECONDS))

  while active_processes_alive && ((SECONDS < deadline)); do
    sleep 0.1
  done
  active_processes_alive || return 0

  for ((index = 0; index < ${#ACTIVE_PIDS[@]}; index += 1)); do
    pid="${ACTIVE_PIDS[$index]}"
    group="${ACTIVE_GROUPS[$index]}"
    [[ "$pid" == "0" ]] && continue
    if [[ "$group" == "1" ]] && identity_matches "$pid" "${ACTIVE_IDENTITIES[$index]}"; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    elif [[ "$group" == "0" ]] && identity_matches "$pid" "${ACTIVE_IDENTITIES[$index]}"; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    while IFS= read -r identity; do
      [[ -n "$identity" ]] || continue
      descendant_pid="${identity%%:*}"
      identity_matches "$descendant_pid" "$identity" || continue
      kill -KILL "$descendant_pid" 2>/dev/null || true
    done <<< "${ACTIVE_DESCENDANTS[$index]}"
  done
}

terminate_active() {
  local pid
  local identity
  local descendant_pid
  local watchdog_pid
  local -a pids=("${ACTIVE_PIDS[@]}")
  ((${#pids[@]} > 0)) || return 0

  local index
  for ((index = 0; index < ${#pids[@]}; index += 1)); do
    pid="${pids[$index]}"
    [[ "$pid" == "0" ]] && continue
    ACTIVE_DESCENDANTS[$index]="$(descendant_identities_of "$pid")"
    if [[ "${ACTIVE_GROUPS[$index]}" == "1" ]] && identity_matches "$pid" "${ACTIVE_IDENTITIES[$index]}"; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    elif [[ "${ACTIVE_GROUPS[$index]}" == "0" ]] && identity_matches "$pid" "${ACTIVE_IDENTITIES[$index]}"; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
    while IFS= read -r identity; do
      [[ -n "$identity" ]] || continue
      descendant_pid="${identity%%:*}"
      identity_matches "$descendant_pid" "$identity" || continue
      kill -TERM "$descendant_pid" 2>/dev/null || true
    done <<< "${ACTIVE_DESCENDANTS[$index]}"
  done
  force_kill_after_grace &
  watchdog_pid=$!
  for pid in "${pids[@]}"; do
    [[ "$pid" == "0" ]] || wait "$pid" 2>/dev/null || true
  done
  wait "$watchdog_pid" 2>/dev/null || true
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  terminate_active
  exit "$status"
}

run_phase() {
  local variant="$1"
  local trials="$2"
  local sampler="$3"
  local total_trials="$4"
  local objective
  local work_dir
  local index
  local active_index
  local launched_pid
  local launched_identity
  local failures=0

  ACTIVE_PIDS=()
  ACTIVE_NAMES=()
  ACTIVE_GROUPS=()
  ACTIVE_IDENTITIES=()
  ACTIVE_DESCENDANTS=()
  echo
  echo "==> $variant ($sampler, $total_trials new trials per objective)"
  if [[ "$trials" != "$total_trials" ]]; then
    echo "    budget: $trials + $REFINE_ROUNDS x $REFINE_TRIALS = $total_trials"
  fi
  for objective in "${OBJECTIVES[@]}"; do
    work_dir="$OUT_ROOT/$objective/$variant"
    mkdir -p -- "$work_dir"
    make_command "$objective" "$variant" "$trials" "$sampler"
    if command -v setsid >/dev/null 2>&1; then
      PATH="$ROOT_DIR/.venv/bin:$PATH" setsid "${COMMAND[@]}" >"$work_dir/run.log" 2>&1 &
      ACTIVE_GROUPS+=(1)
    else
      PATH="$ROOT_DIR/.venv/bin:$PATH" "${COMMAND[@]}" >"$work_dir/run.log" 2>&1 &
      ACTIVE_GROUPS+=(0)
    fi
    launched_pid=$!
    ACTIVE_PIDS+=("$launched_pid")
    ACTIVE_IDENTITIES+=("")
    active_index=$((${#ACTIVE_PIDS[@]} - 1))
    if launched_identity="$(process_identity "$launched_pid")"; then
      ACTIVE_IDENTITIES[$active_index]="$launched_identity"
    else
      wait "$launched_pid" 2>/dev/null || true
      ACTIVE_PIDS[$active_index]=0
      fail "could not track $objective/$variant process identity"
    fi
    ACTIVE_NAMES+=("$objective/$variant")
    echo "    started $objective (pid $launched_pid, log $work_dir/run.log)"
  done

  for ((index = 0; index < ${#ACTIVE_PIDS[@]}; index += 1)); do
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
  validate_configuration
  prepare_cli
  mkdir -p -- "$(dirname "$OUT_ROOT")"
  mkdir -m 700 -- "$OUT_ROOT" 2>/dev/null || fail "output root already exists or cannot be created: $OUT_ROOT"

  run_phase 01_base_cmaes "$TOTAL_TRIALS" cmaes "$TOTAL_TRIALS"
  run_phase 02_reset_no_transfer "$REFINE_INITIAL_TRIALS" cmaes "$TOTAL_TRIALS"
  run_phase 03_reset_with_transfer "$REFINE_INITIAL_TRIALS" cmaes "$TOTAL_TRIALS"
  run_phase 04_centaur "$TOTAL_TRIALS" centaur "$TOTAL_TRIALS"

  echo
  echo "BBOB experiments complete: $OUT_ROOT"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
main "$@"
