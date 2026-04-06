"""
config.py  —  Single source of truth for all paths and settings.

FLOW (matches James's panGPT exactly):
======================================
  raw gene-order.gz
       ↓
  01_preprocess.py
       ↓  produces: pangpt_training_sequences.txt  (one genome per line)
       ↓
  02_split_and_window.py   ← THIS IS WHERE THE SPLIT HAPPENS, EXACTLY LIKE JAMES
       ↓
       ├── split genomes into train / val / test  (80% / 10% / 10%)
       │        using sklearn train_test_split with seed=42
       │        (SAME as panGPT.py does internally)
       │
       ├── apply movingSplits.py logic ONLY on TRAIN genomes
       │        multiple window+stride combos (small → big overlap)
       │
       ├── val genomes   → single fixed window (no augmentation)
       └── test genomes  → single fixed window (no augmentation)
       ↓
  pangpt_train_windows.txt   → fed to panGPT.py  (train_size=1.0, val from separate file)
  pangpt_val_windows.txt
  pangpt_test_windows.txt
       ↓
  03_hyperparam_search.py  (optional)
       ↓
  04_train.py              → calls panGPT.py as-is
       ↓
  02b_phase1_windows.py    → Phase 1 experiments (different window sizes)
       ↓
  05_inference.py
       ↓
  06_anomaly.py

HOW TO PLACE YOUR INPUT DATA:
==============================
  See the bottom of this file and README.md section "Data Setup".
  Short answer: put gene-order.gz into   pangpt_project/data/gene-order.gz
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS  — derived dynamically so they work on any machine / Slurm node
# BUG FIX (Bug 1): removed hardcoded /work/users/tgangil/... paths
# ============================================================================

# Root of this project (where config.py lives)
PROJECT_DIR = Path(__file__).resolve().parent

# Where you cloned panGPT:   git clone https://github.com/mol-evol/panGPT ~/panGPT
PANGPT_DIR  = Path.home() / "panGPT"

# ============================================================================
# DERIVED PATHS  — do not edit
# ============================================================================

DATA_DIR        = PROJECT_DIR / "data"
LOGS_DIR        = PROJECT_DIR / "logs"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR     = PROJECT_DIR / "results"

# Input raw file (place your gene-order.gz here — see README)
DATA_RAW        = DATA_DIR / "gene-order.gz"      # or gene-order (plain text)

# After 01_preprocess.py
DATA_CLEANED_JSON    = DATA_DIR / "cleaned_gene_orders.json"       # genome → gene list
DATA_SEQUENCES_TXT   = DATA_DIR / "pangpt_training_sequences.txt"
DATA_STATS_JSON      = DATA_DIR / "preprocessing_stats.json"

# After 02_split_and_window.py
DATA_TRAIN_WINDOWS   = DATA_DIR / "pangpt_train_windows.txt"   # windowed train → fed to panGPT
DATA_VAL_WINDOWS     = DATA_DIR / "pangpt_val_windows.txt"     # val (fixed window)
DATA_TEST_WINDOWS    = DATA_DIR / "pangpt_test_windows.txt"    # test (fixed window)
SPLIT_STATS_JSON     = DATA_DIR / "split_and_window_stats.json"

# panGPT artefacts
MODEL_CHECKPOINT     = CHECKPOINTS_DIR / "model_checkpoint.pth"
TOKENIZER_FILE       = PROJECT_DIR    / "pangenome_gpt_tokenizer.json"

# panGPT scripts (never modified)
PANGPT_SCRIPT        = PANGPT_DIR / "panGPT.py"
MOVING_SPLITS_SCRIPT = PANGPT_DIR / "movingSplits.py"   # James's original script
PANPROMPT_SCRIPT     = PANGPT_DIR / "panPrompt.py"

# ============================================================================
# SPLIT SETTINGS  (must match panGPT.py defaults exactly)
# ============================================================================

SPLIT = {
    "train_size": 0.8,   # 80% of genomes → training
    "val_size":   0.1,   # 10% → validation
    # remaining 10% → test  (panGPT computes this as 1 - train - val)
    "seed":       42,    # same seed used everywhere for reproducibility
}

# ============================================================================
# WINDOWING SETTINGS
# (applied ONLY to training genomes — val/test use a single fixed window)
# ============================================================================

WINDOW_STRATEGIES = [
    # (window_size, shift_size, label)
    # overlap = 1 - shift/window
    # BUG FIX (Bug 2): corrected label from "no_overlap_25pct" → "no_overlap_0pct"
    (1024, 1024, "no_overlap_0pct"),       # 0%  overlap  — each gene seen once
    (1024,  768, "light_overlap_25pct"),   # 25% overlap
    (1024,  512, "standard_50pct"),        # 50% overlap  — James's original setting
    (1024,  256, "dense_75pct"),           # 75% overlap  — more context variety
    (1024,  128, "very_dense_87pct"),      # 87% overlap  — maximum context density
]

# Fixed window used for val and test (no augmentation)
VAL_TEST_WINDOW_SIZE  = 1024
VAL_TEST_SHIFT_SIZE   = 1024   # no overlap — each region seen exactly once

# Skip genomes shorter than this (can't make even one window)
MIN_GENOME_LENGTH = 1024

# PAD token used when a genome is shorter than the window size in Phase 1
PAD_TOKEN = "[PAD]"

# ============================================================================
# PHASE 1 — WINDOW SIZE EXPERIMENTS
# ============================================================================
# Each entry trains a completely separate model with one clean window size.
# No overlap (shift = window_size) so experiments are clean and comparable.
# Gene order is NEVER violated — remainder tail gets its own end-anchored
# window, and short genomes are padded with PAD_TOKEN on the RIGHT.
#
# Experiments run as a Slurm array job — one GPU job per window size.
# Output datasets land in:  data/phase1/win<N>/train_windows.txt  etc.
# ============================================================================

PHASE1_EXPERIMENTS = [
    # (window_size, label)
    ( 128,  "win128"),
    ( 256,  "win256"),
    ( 512,  "win512"),
    (1024, "win1024"),   # matches original run — acts as baseline
    (2048, "win2048"),
]

# ============================================================================
# TRAINING HYPERPARAMETERS
# (used by 04_train.py; overridden by 03_hyperparam_search.py if you run it)
# ============================================================================

TRAIN = {
    "model_type":          "transformer",
    "embed_dim":           512,
    "num_heads":           8,
    "num_layers":          6,
    "max_seq_length":      512,
    "batch_size":          4,
    "epochs":              150,
    "learning_rate":       0.0001,
    "lr_scheduler_factor": 0.5,
    "lr_patience":         15,
    "weight_decay":        0.0001,
    "early_stop_patience": 30,
    "min_delta":           0.001,
    "max_vocab_size":      50000,
    "model_dropout_rate":  0.1,
    # BUG FIX (Bug 3): use 0.999/0.001 consistently — sklearn requires strictly < 1.0
    # run_pipeline.sh now also passes 0.999/0.001 instead of 1.0/0.0
    "train_size":          0.999,
    "val_size":            0.001,
    "seed":                42,
    "pe_max_len":          5000,
    "pe_dropout_rate":     0.1,
}

# ============================================================================
# HYPERPARAMETER SEARCH GRID
# ============================================================================

HYPERPARAM_SEARCH = {
    "embed_dim":          [256, 512],
    "num_layers":         [4, 6, 8],
    "num_heads":          [8],
    "learning_rate":      [0.001, 0.0001, 0.00005],
    "model_dropout_rate": [0.1, 0.2],
    "batch_size":         [4, 8],
    "epochs_per_trial":   10,
    "n_trials":           12,
    "max_vocab_size":     50000,
    "max_seq_length":     512,
}

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

INFERENCE = {
    "prompt_length":   200,
    "num_predictions": 20,
    "temperature":     0.3,
    "top_k":           5,
    "n_windows_eval":  20,
    "greedy":          True,
}

# ============================================================================
# ANOMALY DETECTION SETTINGS
# ============================================================================

ANOMALY = {
    "baseline_n_genomes":    200,
    "threshold_sigma":       3.0,
    "threshold_percentile":  99,
    "modification_types":    ["insertion", "deletion", "substitution"],
    "n_synthetic_anomalies": 50,
    "insertion_lengths":     [1, 5, 10, 50],
    "deletion_lengths":      [1, 5, 10],
    # BUG FIX (Bug 17): added dedicated substitution_lengths so substitution
    # doesn't silently fall back to deletion_lengths
    "substitution_lengths":  [1, 5, 10, 50],
}

# ============================================================================
# ENSURE DIRS EXIST
# ============================================================================

for _d in [DATA_DIR, LOGS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)