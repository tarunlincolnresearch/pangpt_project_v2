#!/bin/bash
# =============================================================================
# run_pipeline.sh  —  Full PanGPT pipeline, end-to-end
#
# Usage:
#   bash run_pipeline.sh                  # run all steps
#   bash run_pipeline.sh --skip-search    # skip hyperparameter search
#   bash run_pipeline.sh --from 3         # start from step 3
#
# BUG FIXES applied:
#   Bug 3:  training now passes --train_size 0.999 --val_size 0.001
#           instead of 1.0 / 0.0 which would crash sklearn's train_test_split
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_SEARCH=0
FROM_STEP=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-search) SKIP_SEARCH=1 ;;
        --from)        FROM_STEP="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
step()    { echo -e "\n${BLUE}━━━ Step $1: $2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
ok()      { echo -e "${GREEN}✅  $1${NC}"; }
err()     { echo -e "${RED}❌  $1${NC}"; exit 1; }
substep() { echo -e "  ${BLUE}▶  $1${NC}"; }

# =============================================================================
# Phase 1 window sizes — must match PHASE1_EXPERIMENTS in config.py
# =============================================================================
WINDOW_SIZES=("win128" "win256" "win512" "win1024" "win2048")
PHASE1_DATA_DIR="data/phase1"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        PanGPT Pipeline — Phase 1 (Window Sizes)     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Start      : $(date)"
echo "  From step  : $FROM_STEP  |  Skip search: $SKIP_SEARCH"
echo "  Experiments: ${WINDOW_SIZES[*]}"
echo ""

# =============================================================================
# STEP 1 — Preprocessing
# =============================================================================
if [[ $FROM_STEP -le 1 ]]; then
    step 1 "Preprocessing — clean data, log * entries"
    python scripts/01_preprocess.py || err "Preprocessing failed"
    ok "Preprocessing done"
fi

# =============================================================================
# STEP 2 — Phase 1 windowing (one dataset per window size)
# =============================================================================
if [[ $FROM_STEP -le 2 ]]; then
    step 2 "Phase 1 windowing — one dataset per window size"
    python scripts/02b_phase1_windows.py || err "Phase 1 windowing failed"
    ok "Phase 1 datasets created for: ${WINDOW_SIZES[*]}"
fi

# =============================================================================
# STEP 3 — Hyperparameter search (optional, runs once on win512 as reference)
# =============================================================================
if [[ $FROM_STEP -le 3 && $SKIP_SEARCH -eq 0 ]]; then
    step 3 "Hyperparameter search (using win512 as reference window)"
    # BUG FIX (Bug 4): 03_hyperparam_search.py now has argparse, so
    # --input_file is actually read instead of silently ignored.
    python scripts/03_hyperparam_search.py \
        --input_file "${PHASE1_DATA_DIR}/win512/train_windows.txt" \
        || err "Hyperparam search failed"
    ok "Hyperparam search done"
elif [[ $SKIP_SEARCH -eq 1 ]]; then
    echo -e "  ${BLUE}Skipping hyperparam search — using config.py defaults${NC}"
fi

# =============================================================================
# STEP 4 — Training — one model per window size
# =============================================================================
if [[ $FROM_STEP -le 4 ]]; then
    step 4 "Training — one model per window size (${#WINDOW_SIZES[@]} experiments)"

    for WIN in "${WINDOW_SIZES[@]}"; do

        substep "Training experiment: ${WIN}"

        TRAIN_FILE="${PHASE1_DATA_DIR}/${WIN}/train_windows.txt"
        VAL_FILE="${PHASE1_DATA_DIR}/${WIN}/val_windows.txt"
        CKPT_DIR="checkpoints/${WIN}"
        LOG_DIR="logs/${WIN}"

        if [[ ! -f "$TRAIN_FILE" ]]; then
            err "Train file not found: ${TRAIN_FILE}. Did Step 2 complete successfully?"
        fi
        if [[ ! -f "$VAL_FILE" ]]; then
            err "Val file not found: ${VAL_FILE}. Did Step 2 complete successfully?"
        fi

        mkdir -p "$CKPT_DIR"
        mkdir -p "$LOG_DIR"

        # BUG FIX (Bug 3): use 0.999/0.001 instead of 1.0/0.0
        # sklearn's train_test_split requires train_size strictly < 1.0.
        python scripts/04_train.py \
            --input_file        "$TRAIN_FILE" \
            --model_save_path   "${CKPT_DIR}/model_checkpoint.pth" \
            --tokenizer_file    "${CKPT_DIR}/tokenizer.json" \
            --log_dir           "$LOG_DIR" \
            --train_size        0.999 \
            --val_size          0.001 \
            || err "Training failed for experiment: ${WIN}"

        ok "Training done for ${WIN} → checkpoint: ${CKPT_DIR}/model_checkpoint.pth"

    done

    ok "All window size experiments trained successfully"
fi

# =============================================================================
# STEP 5 — Inference & evaluation — one run per window size
# =============================================================================
if [[ $FROM_STEP -le 5 ]]; then
    step 5 "Inference & evaluation — one run per window size"

    for WIN in "${WINDOW_SIZES[@]}"; do

        substep "Inference for experiment: ${WIN}"

        TEST_FILE="${PHASE1_DATA_DIR}/${WIN}/test_windows.txt"
        VAL_FILE="${PHASE1_DATA_DIR}/${WIN}/val_windows.txt"
        CKPT_DIR="checkpoints/${WIN}"
        RESULTS_DIR_WIN="results/${WIN}"

        if [[ ! -f "$TEST_FILE" ]]; then
            err "Test file not found: ${TEST_FILE}. Did Step 2 complete successfully?"
        fi

        mkdir -p "$RESULTS_DIR_WIN"

        # BUG FIX (Bug 11): 05_inference.py now has argparse so these args
        # are actually honoured. Previously all were silently ignored and every
        # experiment used the same hardcoded paths from config.py.
        python scripts/05_inference.py \
            --input_file        "$TEST_FILE" \
            --val_file          "$VAL_FILE" \
            --model_save_path   "${CKPT_DIR}/model_checkpoint.pth" \
            --tokenizer_file    "${CKPT_DIR}/tokenizer.json" \
            --results_dir       "$RESULTS_DIR_WIN" \
            || err "Inference failed for experiment: ${WIN}"

        ok "Inference done for ${WIN} → results: ${RESULTS_DIR_WIN}"

    done

    ok "All inference runs complete"
fi

# =============================================================================
# STEP 6 — Anomaly detection — one run per window size
# =============================================================================
if [[ $FROM_STEP -le 6 ]]; then
    step 6 "Anomaly detection — one run per window size"

    for WIN in "${WINDOW_SIZES[@]}"; do

        substep "Anomaly detection for experiment: ${WIN}"

        TEST_FILE="${PHASE1_DATA_DIR}/${WIN}/test_windows.txt"
        CKPT_DIR="checkpoints/${WIN}"
        RESULTS_DIR_WIN="results/${WIN}"

        mkdir -p "$RESULTS_DIR_WIN"

        python scripts/06_anomaly.py \
            --model_save_path   "${CKPT_DIR}/model_checkpoint.pth" \
            --tokenizer_file    "${CKPT_DIR}/tokenizer.json" \
            --results_dir       "$RESULTS_DIR_WIN" \
            --test_file         "$TEST_FILE" \
            || err "Anomaly detection failed for experiment: ${WIN}"

        ok "Anomaly detection done for ${WIN}"

    done

    ok "All anomaly detection runs complete"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           Phase 1 Pipeline Complete                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  End : $(date)"
echo ""
echo "  Results per window size:"
for WIN in "${WINDOW_SIZES[@]}"; do
    echo "    ${WIN}  →  results/${WIN}/   checkpoints/${WIN}/"
done
echo ""
echo "  Logs      → logs/<window_size>/"
echo "  Summary   → ${PHASE1_DATA_DIR}/phase1_summary.json"