#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="runs/feature_ablation_${RUN_STAMP}"
START_FROM="${START_FROM:-}"
_start_reached="${START_FROM:+0}"
VALID_EXPERIMENTS=(
    early_ablation_mert_only
    early_ablation_mert_mel
    early_ablation_mert_mel_chroma
    early_ablation_full
)

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "Missing required command: $1"
        exit 1
    fi
}

has_cached_features() {
    local cache_dir="$1"
    find "$cache_dir" -type f -name '*.pt' -print -quit 2>/dev/null | grep -q .
}

ensure_precomputed_features() {
    if has_cached_features "data/features/deam" && has_cached_features "data/features/mtg"; then
        log "Precomputed features detected. Skipping feature extraction."
        return
    fi

    log "Precomputed features not found. Running feature extraction first."
    "$PYTHON_BIN" precompute_features.py --dataset all
}

archive_existing_artifacts() {
    local backup_dir="$SUITE_DIR/preexisting"

    mkdir -p "$backup_dir"

    if [[ -d "runs/unified_mood_model_early" ]]; then
        mv "runs/unified_mood_model_early" "$backup_dir/unified_mood_model_early"
        log "Archived existing runs/unified_mood_model_early"
    fi

    if [[ -f "unified_mood_model_early.pt" ]]; then
        mv "unified_mood_model_early.pt" "$backup_dir/unified_mood_model_early.pt"
        log "Archived existing unified_mood_model_early.pt"
    fi
}

save_command_manifest() {
    local dest_dir="$1"
    shift

    {
        printf 'cwd=%s\n' "$ROOT_DIR"
        printf 'command='
        printf '%q ' "$@"
        printf '\n'
    } > "$dest_dir/COMMAND.txt"
}

append_suite_manifest() {
    local experiment_name="$1"
    shift

    {
        printf '[%s]\n' "$experiment_name"
        printf 'cwd=%s\n' "$ROOT_DIR"
        printf 'command='
        printf '%q ' "$@"
        printf '\n\n'
    } >> "$SUITE_DIR/COMMANDS.txt"
}

validate_start_from() {
    local candidate="$1"

    if [[ -z "$candidate" ]]; then
        return 0
    fi

    local experiment
    for experiment in "${VALID_EXPERIMENTS[@]}"; do
        if [[ "$candidate" == "$experiment" ]]; then
            return 0
        fi
    done

    log "Invalid START_FROM=${candidate}"
    log "Allowed values: ${VALID_EXPERIMENTS[*]}"
    exit 1
}

collect_outputs() {
    local experiment_name="$1"
    local dest_dir="$SUITE_DIR/${experiment_name}"

    mkdir -p "$dest_dir"

    if [[ -d "runs/unified_mood_model_early" ]]; then
        mv "runs/unified_mood_model_early" "$dest_dir/"
    fi

    if [[ -f "unified_mood_model_early.pt" ]]; then
        mv "unified_mood_model_early.pt" "$dest_dir/"
    fi
}

run_experiment() {
    local experiment_name="$1"
    local enabled_features="$2"

    if [[ "$_start_reached" == "0" ]]; then
        if [[ "$experiment_name" == "$START_FROM" ]]; then
            _start_reached=1
        else
            log "Skipping experiment: ${experiment_name} (before START_FROM=${START_FROM})"
            return 0
        fi
    fi

    archive_existing_artifacts

    local -a cmd=(
        "$PYTHON_BIN" train.py
        --fusion_type early
        --batch_size "$BATCH_SIZE"
        --epochs "$EPOCHS"
        --num_workers "$NUM_WORKERS"
        --enabled_features "$enabled_features"
    )

    log "Starting experiment: ${experiment_name}"
    append_suite_manifest "$experiment_name" "${cmd[@]}"
    "${cmd[@]}"
    collect_outputs "$experiment_name"
    save_command_manifest "$SUITE_DIR/${experiment_name}" "${cmd[@]}"
    log "Finished experiment: ${experiment_name}"
}

require_command "$PYTHON_BIN"
validate_start_from "$START_FROM"
mkdir -p "$SUITE_DIR"

log "Feature ablation output directory: $SUITE_DIR"
ensure_precomputed_features

run_experiment early_ablation_mert_only mert
run_experiment early_ablation_mert_mel mert,mel
run_experiment early_ablation_mert_mel_chroma mert,mel,chroma
run_experiment early_ablation_full mert,mel,chroma,tempogram

log "All feature ablation experiments finished. Collected outputs under: $SUITE_DIR"
