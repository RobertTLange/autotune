#!/usr/bin/env bash
#SBATCH --job-name=autotune-cifar10-speedrun
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 077

if [[ -z "${ROOT_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/package.json" ]]; then
        ROOT_DIR="$SLURM_SUBMIT_DIR"
    else
        ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
if [[ ! -f "$ROOT_DIR/package.json" || ! -d "$ROOT_DIR/examples" ]]; then
    echo "Could not locate autotune repo root at ROOT_DIR=$ROOT_DIR" >&2
    echo "Submit from the repo root or set ROOT_DIR=/path/to/autotune." >&2
    exit 2
fi
ROOT_DIR="$(cd "$ROOT_DIR" && pwd -P)"

EXAMPLE_DIR="$ROOT_DIR/examples/cifar10_speedrun"
SCRIPT="$EXAMPLE_DIR/cifar10_speedrun.py"
AGENT="${AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
AGENT_GUIDANCE="${AGENT_GUIDANCE:-Tune only training hyperparameters and preserve the CIFAR-10 speedrun scoring protocol. Do not tune CIFAR10_SPEEDRUN_* environment controls or data/cache paths.}"
SAMPLER="${SAMPLER:-tpe}"
SAMPLER_SEED="${SAMPLER_SEED:-42}"
PRUNER="${PRUNER:-none}"
N_JOBS="${N_JOBS:-1}"

validate_decimal_integer() {
    local name="$1"
    local value="$2"
    local minimum="$3"
    local maximum="$4"
    if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ || ${#value} -gt ${#maximum} ]]; then
        echo "$name must be a decimal integer between $minimum and $maximum" >&2
        exit 2
    fi
    if ((10#$value < minimum || 10#$value > maximum)); then
        echo "$name must be a decimal integer between $minimum and $maximum" >&2
        exit 2
    fi
}

validate_private_parent_chain() {
    local target="$1"
    local current mode owner permissions
    current="$(dirname "$target")"
    while [[ "$current" != "/" ]]; do
        mode="$(stat -c %a "$current")"
        owner="$(stat -c %u "$current")"
        permissions=$((8#$mode))
        if ((permissions & 0022)); then
            if ! ((permissions & 01000)) || [[ "$owner" != "0" && "$owner" != "$(id -u)" ]]; then
                echo "Path has an unsafe writable parent: $current" >&2
                exit 2
            fi
        fi
        current="$(dirname "$current")"
    done
}

BASELINE_TRIALS="${BASELINE_TRIALS:-${SINGLE_SHOT_TRIALS:-500}}"
REFINE_INITIAL_TRIALS="${REFINE_INITIAL_TRIALS:-125}"
REFINE_ROUNDS="${REFINE_ROUNDS:-3}"
REFINE_TRIALS="${REFINE_TRIALS:-125}"
CENTAUR_TRIALS="${CENTAUR_TRIALS:-$BASELINE_TRIALS}"
ALLOW_UNEQUAL_TRIALS="${ALLOW_UNEQUAL_TRIALS:-0}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-162000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
CANCEL_GRACE_SECONDS="${CANCEL_GRACE_SECONDS:-5}"
CENTAUR_LLM_PROBABILITY="${CENTAUR_LLM_PROBABILITY:-0.3}"
CENTAUR_WARMUP_TRIALS="${CENTAUR_WARMUP_TRIALS:-10}"
CENTAUR_SEED="${CENTAUR_SEED:-42}"

validate_decimal_integer BASELINE_TRIALS "$BASELINE_TRIALS" 1 1000000
validate_decimal_integer REFINE_INITIAL_TRIALS "$REFINE_INITIAL_TRIALS" 1 1000000
validate_decimal_integer REFINE_ROUNDS "$REFINE_ROUNDS" 0 1000
validate_decimal_integer REFINE_TRIALS "$REFINE_TRIALS" 1 1000000
validate_decimal_integer CENTAUR_TRIALS "$CENTAUR_TRIALS" 1 1000000
validate_decimal_integer TIME_BUDGET_SECONDS "$TIME_BUDGET_SECONDS" 1 172800
validate_decimal_integer TIMEOUT_SECONDS "$TIMEOUT_SECONDS" 1 86400
validate_decimal_integer CANCEL_GRACE_SECONDS "$CANCEL_GRACE_SECONDS" 0 300
validate_decimal_integer SAMPLER_SEED "$SAMPLER_SEED" 0 2147483647
validate_decimal_integer CENTAUR_WARMUP_TRIALS "$CENTAUR_WARMUP_TRIALS" 0 1000000
validate_decimal_integer CENTAUR_SEED "$CENTAUR_SEED" 0 2147483647
if [[ "$ALLOW_UNEQUAL_TRIALS" != "0" && "$ALLOW_UNEQUAL_TRIALS" != "1" ]]; then
    echo "ALLOW_UNEQUAL_TRIALS must be 0 or 1" >&2
    exit 2
fi
if [[ "$N_JOBS" != "1" ]]; then
    echo "N_JOBS must be 1 for the one-GPU comparable protocol" >&2
    exit 2
fi
if [[ "$SAMPLER" == "centaur" ]]; then
    echo "SAMPLER must be a non-Centaur Optuna sampler; Centaur has a dedicated fourth arm" >&2
    exit 2
fi
if [[ ! "$CENTAUR_LLM_PROBABILITY" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
    echo "CENTAUR_LLM_PROBABILITY must be between 0 and 1" >&2
    exit 2
fi

REFINED_TOTAL_TRIALS=$((REFINE_INITIAL_TRIALS + REFINE_ROUNDS * REFINE_TRIALS))
if [[ "$ALLOW_UNEQUAL_TRIALS" != "1" &&
    ("$BASELINE_TRIALS" -ne "$REFINED_TOTAL_TRIALS" || "$BASELINE_TRIALS" -ne "$CENTAUR_TRIALS") ]]; then
    echo "Expected equal trial budgets; baseline=$BASELINE_TRIALS refined=$REFINED_TOTAL_TRIALS centaur=$CENTAUR_TRIALS" >&2
    echo "Set ALLOW_UNEQUAL_TRIALS=1 to run unequal budgets." >&2
    exit 2
fi

export CIFAR10_SPEEDRUN_NUM_RUNS="${CIFAR10_SPEEDRUN_NUM_RUNS:-100}"
export CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN="${CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN:-3.0}"
export CIFAR10_SPEEDRUN_DATA_DIR="${CIFAR10_SPEEDRUN_DATA_DIR:-$HOME/.cache/autotune/cifar10}"
validate_decimal_integer CIFAR10_SPEEDRUN_NUM_RUNS "$CIFAR10_SPEEDRUN_NUM_RUNS" 1 10000
if [[ ! "$CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN" =~ ^(0|[1-9][0-9]{0,3})(\.[0-9]+)?$ ||
    "$CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN" =~ ^0(\.0+)?$ ]]; then
    echo "CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN must be greater than 0 and less than 10000" >&2
    exit 2
fi

RUN_GROUP="${RUN_GROUP:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
if [[ ! "$RUN_GROUP" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_GROUP may contain only letters, numbers, dots, underscores, and hyphens" >&2
    exit 2
fi
OUT_ROOT="${OUT_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/autotune/cifar10_speedrun_ablations/$RUN_GROUP}"
CACHE_ROOT="${AUTOTUNE_RUNTIME_CACHE_DIR:-$OUT_ROOT/runtime_cache}"
INITIAL_CONFIG="$OUT_ROOT/search_space.initial.yaml"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "nvidia-smi failed; continuing so the workload can report the real CUDA error if needed." >&2
fi
if [[ -n "${LOAD_MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load $LOAD_MODULES
fi
if [[ -n "${SETUP_COMMAND:-}" ]]; then
    # Trusted operator hook; it intentionally runs in this shell so environment changes persist.
    eval -- "$SETUP_COMMAND"
fi

AUTOTUNE_PYTHON="${AUTOTUNE_PYTHON:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$AUTOTUNE_PYTHON" ]]; then
    AUTOTUNE_PYTHON="$(command -v python3)"
fi
AUTOTUNE_PYTHON="$(cd "$(dirname "$AUTOTUNE_PYTHON")" && pwd -P)/$(basename "$AUTOTUNE_PYTHON")"
AUTOTUNE_PYTHON_DIR="$(dirname "$AUTOTUNE_PYTHON")"
RUNTIME_PATH="$AUTOTUNE_PYTHON_DIR:$PATH"
"$AUTOTUNE_PYTHON" -c "import torch, torchvision"

if [[ -L "$OUT_ROOT" ]]; then
    echo "OUT_ROOT must not be a symlink: $OUT_ROOT" >&2
    exit 2
fi
OUT_ROOT="$(realpath -m "$OUT_ROOT")"
if [[ -L "$CACHE_ROOT" ]]; then
    echo "AUTOTUNE_RUNTIME_CACHE_DIR must not be a symlink: $CACHE_ROOT" >&2
    exit 2
fi
CACHE_ROOT="$(realpath -m "$CACHE_ROOT")"
INITIAL_CONFIG="$(realpath -m "$INITIAL_CONFIG")"
export CIFAR10_SPEEDRUN_DATA_DIR="$(realpath -m "$CIFAR10_SPEEDRUN_DATA_DIR")"
if [[ "$OUT_ROOT" == *[?#%]* ]]; then
    echo "OUT_ROOT may not contain SQLite URL metacharacters (?, #, or %): $OUT_ROOT" >&2
    exit 2
fi
DISCOVERY_MANIFEST="$OUT_ROOT/discovery_complete.json"
mkdir -p "$(dirname "$OUT_ROOT")"
validate_private_parent_chain "$OUT_ROOT"
if ! mkdir -m 700 "$OUT_ROOT"; then
    echo "Run root already exists; choose a new RUN_GROUP or OUT_ROOT: $OUT_ROOT" >&2
    exit 2
fi
mkdir -p -m 700 "$CACHE_ROOT"
validate_private_parent_chain "$CACHE_ROOT"
if [[ ! -d "$CACHE_ROOT" || "$(stat -c %u "$CACHE_ROOT")" != "$(id -u)" ]]; then
    echo "Runtime cache must be an owned directory: $CACHE_ROOT" >&2
    exit 2
fi
chmod 700 "$CACHE_ROOT"

cd "$ROOT_DIR"
npm run build
"$AUTOTUNE_PYTHON" "$EXAMPLE_DIR/prepare_cifar10_data.py" \
    --data-dir "$CIFAR10_SPEEDRUN_DATA_DIR"

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
PATH="$RUNTIME_PATH" "${analyze_cmd[@]}"

describe_variant() {
    local name="$1"
    local work_dir="$OUT_ROOT/$name"
    local study_name="cifar10_${RUN_GROUP}_${name}"
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
    local trials="$2"
    local sampler="$3"
    shift 3
    local work_dir="$OUT_ROOT/$name"
    local cache_dir="$CACHE_ROOT/$name"
    local study_name="cifar10_${RUN_GROUP}_${name}"
    local storage="sqlite:///$work_dir/study.db"
    local -a cmd=(
        node "$ROOT_DIR/dist/cli.js" run "$SCRIPT"
        --config "$INITIAL_CONFIG"
        --trials "$trials"
        --sampler "$sampler"
        --pruner "$PRUNER"
        --n-jobs "$N_JOBS"
        --timeout-seconds "$TIMEOUT_SECONDS"
        --time-budget-seconds "$TIME_BUDGET_SECONDS"
        --storage "$storage"
        --study-name "$study_name"
        --direction maximize
        --agent "$AGENT"
        --agent-guidance "$AGENT_GUIDANCE"
        --work-dir "$work_dir"
        --yes
        --json
        "${common_agent_flags[@]}"
    )
    if [[ "$sampler" != "centaur" ]]; then
        cmd+=(--sampler-seed "$SAMPLER_SEED")
    fi

    mkdir -p \
        "$cache_dir/tmp" \
        "$cache_dir/torchinductor" \
        "$cache_dir/triton" \
        "$cache_dir/cuda" \
        "$cache_dir/torch_extensions" \
        "$cache_dir/xdg" \
        "$work_dir/trial_metrics"
    exec srun --exclusive --ntasks=1 --cpus-per-task=8 --gres=gpu:1 /usr/bin/env \
        TMPDIR="$cache_dir/tmp" \
        TORCHINDUCTOR_CACHE_DIR="$cache_dir/torchinductor" \
        TRITON_CACHE_DIR="$cache_dir/triton" \
        CUDA_CACHE_PATH="$cache_dir/cuda" \
        TORCH_EXTENSIONS_DIR="$cache_dir/torch_extensions" \
        XDG_CACHE_HOME="$cache_dir/xdg" \
        CIFAR10_SPEEDRUN_RESULTS_DIR="$work_dir/trial_metrics" \
        PATH="$RUNTIME_PATH" \
        "${cmd[@]}" "$@"
}

active_variant_pids=()
declare -A variant_names_by_pid=()
declare -A variant_logs_by_pid=()

launch_variant() {
    local name="$1"
    local work_dir="$OUT_ROOT/$name"
    describe_variant "$name"
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

verify_variant_completion() {
    local name="$1"
    local round expected_trials
    local -a results=()
    case "$name" in
    01_optuna_baseline)
        results+=(--result "optuna_baseline=$BASELINE_TRIALS:$OUT_ROOT/$name/results.json")
        ;;
    04_centaur)
        results+=(--result "centaur=$CENTAUR_TRIALS:$OUT_ROOT/$name/results.json")
        ;;
    02_autotune_resets_no_trial_transfer | 03_autotune_resets_trial_transfer)
        for ((round = 0; round <= REFINE_ROUNDS; round++)); do
            expected_trials="$REFINE_TRIALS"
            [[ "$round" -eq 0 ]] && expected_trials="$REFINE_INITIAL_TRIALS"
            results+=(--result "${name}_round_$round=$expected_trials:$OUT_ROOT/$name/results.round_$round.json")
        done
        ;;
    *)
        echo "Unknown discovery arm: $name" >&2
        return 2
        ;;
    esac
    "$AUTOTUNE_PYTHON" "$EXAMPLE_DIR/manage_cifar10_run.py" check --root "$OUT_ROOT" "${results[@]}"
}

trap cancel_active_variants EXIT
trap 'exit 130' INT TERM

wait_for_variants() {
    local finished_pid status name log
    while ((${#active_variant_pids[@]} > 0)); do
        finished_pid=""
        if wait -n -p finished_pid "${active_variant_pids[@]}"; then
            status=0
        else
            status="$?"
        fi
        if [[ -z "$finished_pid" ]]; then
            echo "Unable to identify the completed CIFAR-10 discovery arm" >&2
            cancel_active_variants
            return 1
        fi
        name="${variant_names_by_pid[$finished_pid]}"
        log="${variant_logs_by_pid[$finished_pid]}"
        remove_active_variant "$finished_pid"
        if [[ "$status" -eq 0 ]]; then
            verify_variant_completion "$name" || status="$?"
        fi
        if [[ "$status" -eq 0 ]]; then
            echo "<== completed $name"
            continue
        fi
        echo "<== failed $name (exit $status; log $log)" >&2
        tail -n 40 "$log" >&2 || true
        cancel_active_variants
        return 1
    done
}

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
"$AUTOTUNE_PYTHON" "$EXAMPLE_DIR/manage_cifar10_run.py" create \
    --manifest "$DISCOVERY_MANIFEST" \
    --refine-rounds "$REFINE_ROUNDS" \
    "${manifest_results[@]}"

echo
echo "CIFAR-10 speedrun ablations complete."
echo "Results root: $OUT_ROOT"
echo "Discovery manifest: $DISCOVERY_MANIFEST"
