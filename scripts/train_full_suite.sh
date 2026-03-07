#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="runs/experiment_suite_${RUN_STAMP}"

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
    local fusion_type="$1"
    local backup_dir="$SUITE_DIR/preexisting/${fusion_type}"

    mkdir -p "$backup_dir"

    if [[ -d "runs/unified_mood_model_${fusion_type}" ]]; then
        mv "runs/unified_mood_model_${fusion_type}" "$backup_dir/unified_mood_model_${fusion_type}"
        log "Archived existing runs/unified_mood_model_${fusion_type}"
    fi

    if [[ -f "unified_mood_model_${fusion_type}.pt" ]]; then
        mv "unified_mood_model_${fusion_type}.pt" "$backup_dir/unified_mood_model_${fusion_type}.pt"
        log "Archived existing unified_mood_model_${fusion_type}.pt"
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

collect_outputs() {
    local experiment_name="$1"
    local fusion_type="$2"
    local dest_dir="$SUITE_DIR/${experiment_name}"

    mkdir -p "$dest_dir"

    if [[ -d "runs/unified_mood_model_${fusion_type}" ]]; then
        mv "runs/unified_mood_model_${fusion_type}" "$dest_dir/"
    fi

    if [[ -f "unified_mood_model_${fusion_type}.pt" ]]; then
        mv "unified_mood_model_${fusion_type}.pt" "$dest_dir/"
    fi
}

run_experiment() {
    local experiment_name="$1"
    local fusion_type="$2"
    shift 2

    archive_existing_artifacts "$fusion_type"

    local -a cmd=(
        "$PYTHON_BIN" train.py
        --fusion_type "$fusion_type"
        --batch_size "$BATCH_SIZE"
        --epochs "$EPOCHS"
        --num_workers "$NUM_WORKERS"
    )
    cmd+=("$@")

    log "Starting experiment: ${experiment_name}"
    save_command_manifest "$SUITE_DIR" "${cmd[@]}"
    "${cmd[@]}"
    collect_outputs "$experiment_name" "$fusion_type"
    save_command_manifest "$SUITE_DIR/${experiment_name}" "${cmd[@]}"
    log "Finished experiment: ${experiment_name}"
}

require_command "$PYTHON_BIN"
mkdir -p "$SUITE_DIR"

log "Experiment suite output directory: $SUITE_DIR"
ensure_precomputed_features

run_experiment early_baseline early
run_experiment late_baseline late

# Extra ablations currently supported by train.py without changing Python code.
run_experiment early_norm_deam_labels early --normalize_deam_labels
run_experiment late_norm_deam_labels late --normalize_deam_labels

log "All experiments finished. Collected outputs under: $SUITE_DIR"
log "You can inspect TensorBoard logs inside each experiment directory."