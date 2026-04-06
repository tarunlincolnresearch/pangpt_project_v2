#!/usr/bin/env python3
"""
02_split_and_window.py
======================
This script does TWO things in the correct order:

  STEP A — Split genomes into train / val / test
  -----------------------------------------------
  Uses sklearn.model_selection.train_test_split with seed=42.
  This is EXACTLY what James's panGPT.py does internally.
  We do it here at the GENOME level (before windowing) so that:
    - val and test genomes are never seen during training
    - windowing cannot leak test data into training windows

  STEP B — Apply moving window ONLY on training genomes
  ------------------------------------------------------
  Uses James's exact movingSplits.py logic:
      [genome[i : i+window_size] for i in range(0, len-window+1, shift)]

  We run this with MULTIPLE (window, shift) pairs — from NO overlap to
  MAXIMUM overlap — to give the model richer context variety.

  Val and test genomes get ONE fixed window pass (no augmentation).

OUTPUT FILES
------------
  data/pangpt_train_windows.txt  ← windowed train data  → this goes into panGPT
  data/pangpt_val_windows.txt    ← val data (single fixed window)
  data/pangpt_test_windows.txt   ← test data (single fixed window)

WHY THIS ORDER MATTERS
-----------------------
In the original notebook, windowing happened BEFORE any split, then
panGPT.py split the windows internally. That means windows from the
SAME genome could end up in both train AND val/test — a data leakage
problem. Splitting genomes first and then windowing only the train set
eliminates this entirely.
"""

import sys
import json
import random
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_SEQUENCES_TXT,
    DATA_TRAIN_WINDOWS, DATA_VAL_WINDOWS, DATA_TEST_WINDOWS,
    SPLIT_STATS_JSON, LOGS_DIR,
    SPLIT, WINDOW_STRATEGIES,
    VAL_TEST_WINDOW_SIZE, VAL_TEST_SHIFT_SIZE,
    MIN_GENOME_LENGTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "02_split_and_window.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# JAMES'S EXACT movingSplits.py LOGIC
# Source: https://github.com/mol-evol/panGPT/blob/main/movingSplits.py
#
# def moving_window_split(genome, window_size, shift_size):
#     return [genome[i:i + window_size]
#             for i in range(0, len(genome) - window_size + 1, shift_size)]
# ============================================================================

def moving_window_split(genome: list, window_size: int, shift_size: int) -> list:
    """
    James's exact moving window function from movingSplits.py.
    genome     : list of gene name strings
    window_size: number of genes per window
    shift_size : step between window starts
    returns    : list of windows (each window = list of gene strings)
    """
    return [
        genome[i : i + window_size]
        for i in range(0, len(genome) - window_size + 1, shift_size)
    ]


# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("▶  Step 2 — Split Genomes → Apply Moving Window on Train Only")
    log.info("   Input : %s", DATA_SEQUENCES_TXT)

    if not Path(DATA_SEQUENCES_TXT).exists():
        log.error("Input not found. Run 01_preprocess.py first.")
        sys.exit(1)

    # ── LOAD ALL GENOMES ─────────────────────────────────────────────────────
    log.info("Loading genome sequences …")
    all_genomes = []
    skipped_short = 0

    with open(DATA_SEQUENCES_TXT) as fh:
        for line in fh:
            genes = line.strip().split()
            if len(genes) >= MIN_GENOME_LENGTH:
                all_genomes.append(genes)
            else:
                skipped_short += 1

    log.info("  Genomes loaded   : %d", len(all_genomes))
    log.info("  Skipped (too short, < %d genes) : %d", MIN_GENOME_LENGTH, skipped_short)

    # ── STEP A: SPLIT AT GENOME LEVEL ────────────────────────────────────────
    # Exactly mirrors what panGPT.py does:
    #   train_data, temp_data = train_test_split(sequences, train_size=train_size, random_state=seed)
    #   val_data, test_data   = train_test_split(temp_data, train_size=val_ratio, random_state=seed)
    # We replicate this logic precisely.

    train_size = SPLIT["train_size"]   # 0.8
    val_size   = SPLIT["val_size"]     # 0.1
    seed       = SPLIT["seed"]         # 42

    # val_ratio is val_size expressed as a fraction of the remaining (non-train) data
    val_ratio  = val_size / (1.0 - train_size)   # 0.1 / 0.2 = 0.5

    log.info("")
    log.info("STEP A — Genome-level split (mirrors panGPT.py exactly)")
    log.info("  train_size = %.0f%%   val_size = %.0f%%   test_size = %.0f%%",
             train_size * 100, val_size * 100, (1 - train_size - val_size) * 100)
    log.info("  random seed = %d", seed)

    train_genomes, temp_genomes = train_test_split(
        all_genomes,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
    )
    val_genomes, test_genomes = train_test_split(
        temp_genomes,
        train_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    log.info("  Train genomes : %d", len(train_genomes))
    log.info("  Val genomes   : %d", len(val_genomes))
    log.info("  Test genomes  : %d", len(test_genomes))

    # ── STEP B: MULTI-STRATEGY WINDOWING ON TRAIN ONLY ───────────────────────
    log.info("")
    log.info("STEP B — Multi-strategy moving window (TRAIN set only)")
    log.info("  Using James's exact moving_window_split logic")
    log.info("  Strategies ordered from NO overlap → MAXIMUM overlap:")

    all_train_windows = []
    strategy_stats = {}

    for (window_size, shift_size, label) in WINDOW_STRATEGIES:
        overlap_pct = round(100.0 * (1.0 - shift_size / window_size), 1)

        log.info("")
        log.info("  Strategy: %-25s  window=%d  shift=%d  overlap=%.0f%%",
                 label, window_size, shift_size, overlap_pct)

        strategy_windows = []
        genome_counts = []

        for genome in train_genomes:
            windows = moving_window_split(genome, window_size, shift_size)
            strategy_windows.extend(windows)
            genome_counts.append(len(windows))

        log.info("    Windows generated : %d", len(strategy_windows))
        if genome_counts:
            log.info("    Per genome — min=%d  max=%d  avg=%.1f",
                     min(genome_counts), max(genome_counts),
                     sum(genome_counts) / len(genome_counts))

        strategy_stats[label] = {
            "window_size":   window_size,
            "shift_size":    shift_size,
            "overlap_pct":   overlap_pct,
            "windows_count": len(strategy_windows),
        }

        all_train_windows.extend(strategy_windows)

    # Shuffle the combined training windows (prevents ordering bias)
    random.seed(seed)
    random.shuffle(all_train_windows)
    log.info("")
    log.info("  Total training windows (all strategies combined) : %d", len(all_train_windows))
    log.info("  Shuffled with seed=%d ✓", seed)

    # ── VAL AND TEST: SINGLE FIXED WINDOW (NO AUGMENTATION) ──────────────────
    log.info("")
    log.info("STEP C — Fixed window for Val and Test (no augmentation)")
    log.info("  window=%d  shift=%d  (no overlap)", VAL_TEST_WINDOW_SIZE, VAL_TEST_SHIFT_SIZE)

    val_windows = []
    for genome in val_genomes:
        val_windows.extend(moving_window_split(genome, VAL_TEST_WINDOW_SIZE, VAL_TEST_SHIFT_SIZE))

    test_windows = []
    for genome in test_genomes:
        test_windows.extend(moving_window_split(genome, VAL_TEST_WINDOW_SIZE, VAL_TEST_SHIFT_SIZE))

    log.info("  Val windows  : %d", len(val_windows))
    log.info("  Test windows : %d", len(test_windows))

    # ── WRITE OUTPUT FILES ────────────────────────────────────────────────────
    log.info("")
    log.info("Writing output files …")

    def write_windows(windows, path):
        with open(path, "w") as fh:
            for window in windows:
                fh.write(" ".join(window) + "\n")
        log.info("  Wrote %d windows → %s", len(windows), path)

    write_windows(all_train_windows, DATA_TRAIN_WINDOWS)
    write_windows(val_windows,       DATA_VAL_WINDOWS)
    write_windows(test_windows,      DATA_TEST_WINDOWS)

    # ── SAVE STATS ────────────────────────────────────────────────────────────
    stats = {
        "total_genomes":           len(all_genomes),
        "skipped_short_genomes":   skipped_short,
        "train_genomes":           len(train_genomes),
        "val_genomes":             len(val_genomes),
        "test_genomes":            len(test_genomes),
        "split": {
            "train_size": train_size,
            "val_size":   val_size,
            "seed":       seed,
        },
        "train_window_strategies": strategy_stats,
        "total_train_windows":     len(all_train_windows),
        "total_val_windows":       len(val_windows),
        "total_test_windows":      len(test_windows),
        "val_test_window":         {"window_size": VAL_TEST_WINDOW_SIZE, "shift_size": VAL_TEST_SHIFT_SIZE},
        "note": (
            "Windows are produced using James McInerney's exact moving_window_split "
            "from movingSplits.py. Genome-level split mirrors panGPT.py's "
            "sklearn.train_test_split logic. Moving window applied ONLY to train genomes."
        ),
    }
    with open(SPLIT_STATS_JSON, "w") as fh:
        json.dump(stats, fh, indent=2)
    log.info("Stats saved → %s", SPLIT_STATS_JSON)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("  Train windows  : %7d  (fed to panGPT.py)", len(all_train_windows))
    log.info("  Val windows    : %7d", len(val_windows))
    log.info("  Test windows   : %7d", len(test_windows))
    log.info("")
    log.info("  NEXT STEP:")
    log.info("  Feed  %s  into panGPT.py", DATA_TRAIN_WINDOWS)
    log.info("  (use --train_size 1.0 --val_size 0.0 in panGPT if you want")
    log.info("   to control val separately, OR let panGPT re-split internally)")
    log.info("✅  Split and windowing complete.")


if __name__ == "__main__":
    main()
