#!/usr/bin/env python3
"""
02b_phase1_windows.py
=====================
Phase 1 experiment — generate ONE clean dataset per window size.

KEY DESIGN DECISIONS
--------------------
1. GENE ORDER IS SACRED — never shuffled, reversed, or reordered within
   any window or across windows. Every window is always a contiguous slice
   of the original genome in its original left-to-right order.

2. REMAINDER WINDOW — when a genome does not divide evenly into the
   window size, the leftover tail is NOT dropped. Instead we take one
   final window anchored to the END of the genome. This window overlaps
   the previous one slightly, but every gene still appears in order and
   nothing is lost.

   Example (genome=4786 genes, window=2048):
     Window 1: genes[0    : 2048]   positions 1–2048
     Window 2: genes[2048 : 4096]   positions 2049–4096
     Window 3: genes[2738 : 4786]   positions 2739–4786  ← end-anchored remainder
                      ^^^^ overlaps Window 2 by 1358 genes, but ORDER is intact

3. SHORT GENOME PADDING — if a genome is shorter than the window size it
   cannot produce even one window. Rather than discarding it, we pad it
   on the RIGHT with [PAD] tokens up to window_size. Gene order of the
   real genes is fully preserved; padding only appends to the right end.

   Example (genome=300 genes, window=512):
     Window: [gene1, gene2, ..., gene300, [PAD], [PAD], ..., [PAD]]
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          212 padding tokens on the right

4. GENOME-LEVEL SPLIT — uses the exact same seed=42 / 80-10-10 split as
   the main pipeline so train/val/test genomes are identical across all
   experiments. No data leakage is possible because the split happens
   before any windowing.

5. ONE EXPERIMENT = ONE CLEAN DATASET — no overlap strategies are mixed.
   Each window size gets its own isolated train/val/test files so model
   performance can be compared fairly across window sizes.

6. NO SHUFFLING — windows are written in the exact order they are
   extracted from each genome. The biological gene order across windows
   is fully preserved. The model sees windows in genome order, which
   reflects the true sequential structure of the genome.

OUTPUT
------
  data/phase1/win128/
      train_windows.txt
      val_windows.txt
      test_windows.txt
      dataset_stats.json

  data/phase1/win256/  ...
  data/phase1/win512/  ...
  data/phase1/win1024/ ...
  data/phase1/win2048/ ...

  data/phase1/phase1_summary.json   ← summary across all experiments
"""

import sys
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_SEQUENCES_TXT,
    LOGS_DIR,
    SPLIT,
    PHASE1_EXPERIMENTS,
    PAD_TOKEN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "02b_phase1_windows.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# WINDOWING LOGIC
# ============================================================================

def make_windows(genome: list, window_size: int) -> list:
    """
    Slice genome into non-overlapping windows of exactly window_size genes.

    Gene order is NEVER violated. Rules:
      - shift = window_size  (no overlap between regular windows)
      - If the genome divides evenly → no remainder window needed
      - If there is a tail (len % window_size != 0) → add one final window
        anchored to the END of the genome. This window may overlap the
        previous one slightly but every gene is in its original order.
      - If genome is shorter than window_size → pad on the RIGHT with
        PAD_TOKEN to reach exactly window_size. Real gene order untouched.

    Parameters
    ----------
    genome      : list of gene-name strings in original order
    window_size : number of genes per window

    Returns
    -------
    list of windows — each window is a list of exactly window_size strings,
    returned in the original genome order (no shuffling).
    """
    n = len(genome)

    # ── Case 1: genome shorter than window — pad on the right ────────────────
    if n < window_size:
        pad_needed = window_size - n
        padded = genome + [PAD_TOKEN] * pad_needed
        # Sanity check: original genes are untouched
        assert padded[:n] == genome, "BUG: padding modified original gene order"
        assert len(padded) == window_size, "BUG: padded window wrong length"
        return [padded]

    windows = []

    # ── Case 2: genome fits exactly — no remainder ───────────────────────────
    if n % window_size == 0:
        for start in range(0, n, window_size):
            w = genome[start : start + window_size]
            assert len(w) == window_size, "BUG: window wrong length"
            windows.append(w)
        return windows

    # ── Case 3: genome has a remainder tail ──────────────────────────────────
    # Regular non-overlapping windows first
    last_full_start = (n // window_size - 1) * window_size
    for start in range(0, last_full_start + window_size, window_size):
        w = genome[start : start + window_size]
        assert len(w) == window_size, "BUG: regular window wrong length"
        windows.append(w)

    # Remainder window: anchored to the very end of the genome
    # Starts at (n - window_size) so it ends exactly at n
    remainder_start = n - window_size
    remainder = genome[remainder_start : n]
    assert len(remainder) == window_size,   "BUG: remainder window wrong length"
    assert remainder[-1] == genome[-1],     "BUG: remainder doesn't end at genome end"
    assert remainder[0]  == genome[remainder_start], "BUG: remainder start mismatch"
    windows.append(remainder)

    return windows


def verify_order(genome: list, windows: list, window_size: int):
    """
    Verify that every window is a genuine contiguous slice of the genome
    in the original order. Raises AssertionError if any violation is found.
    Padding tokens are allowed only at the right end of the last window
    of a short genome.
    """
    n = len(genome)
    for w_idx, window in enumerate(windows):
        assert len(window) == window_size, \
            f"Window {w_idx} has length {len(window)}, expected {window_size}"

        first_gene = window[0]

        if first_gene == PAD_TOKEN:
            continue

        # Find all positions of first_gene in genome
        candidates = [i for i, g in enumerate(genome) if g == first_gene]
        matched = False
        for start_pos in candidates:
            real_genes_in_window = [g for g in window if g != PAD_TOKEN]
            genome_slice = genome[start_pos : start_pos + len(real_genes_in_window)]
            if genome_slice == real_genes_in_window:
                matched = True
                break
        assert matched, \
            f"Window {w_idx} gene order not found in genome — ORDER VIOLATION"


# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("▶  Phase 1 — Generate per-window-size datasets")
    log.info("   PAD token : %s", PAD_TOKEN)
    log.info("   Experiments: %s", [label for _, label in PHASE1_EXPERIMENTS])
    log.info("   NOTE: No shuffling applied — windows written in genome order")

    # ── Load all genomes ──────────────────────────────────────────────────────
    log.info("")
    log.info("Loading genome sequences from %s …", DATA_SEQUENCES_TXT)
    all_genomes = []
    with open(DATA_SEQUENCES_TXT) as fh:
        for line in fh:
            genes = line.strip().split()
            if genes:
                all_genomes.append(genes)

    genome_lengths = [len(g) for g in all_genomes]
    log.info("  Genomes loaded : %d", len(all_genomes))
    log.info("  Length — min:%d  max:%d  mean:%.0f  median:%.0f",
             min(genome_lengths), max(genome_lengths),
             sum(genome_lengths) / len(genome_lengths),
             sorted(genome_lengths)[len(genome_lengths) // 2])

    # ── Genome-level split — identical seed/ratio to main pipeline ────────────
    train_size = SPLIT["train_size"]   # 0.8
    val_size   = SPLIT["val_size"]     # 0.1
    seed       = SPLIT["seed"]         # 42
    val_ratio  = val_size / (1.0 - train_size)   # 0.5

    log.info("")
    log.info("Genome-level split (seed=%d, 80/10/10) …", seed)

    train_genomes, temp_genomes = train_test_split(
        all_genomes, train_size=train_size, random_state=seed, shuffle=True
    )
    val_genomes, test_genomes = train_test_split(
        temp_genomes, train_size=val_ratio, random_state=seed, shuffle=True
    )

    log.info("  Train: %d  Val: %d  Test: %d genomes",
             len(train_genomes), len(val_genomes), len(test_genomes))

    # ── Output root ───────────────────────────────────────────────────────────
    base_dir = Path(DATA_SEQUENCES_TXT).parent / "phase1"
    base_dir.mkdir(exist_ok=True)

    overall_summary = {}

    # ── One experiment per window size ────────────────────────────────────────
    for (window_size, label) in PHASE1_EXPERIMENTS:

        log.info("")
        log.info("━" * 65)
        log.info("  Experiment : %s   window_size = %d", label, window_size)
        log.info("━" * 65)

        exp_dir = base_dir / label
        exp_dir.mkdir(exist_ok=True)

        stats = {
            "window_size":   window_size,
            "shift":         window_size,
            "overlap":       "0% (plus one end-anchored remainder window per genome)",
            "pad_token":     PAD_TOKEN,
            "shuffled":      False,   # explicitly recorded
        }

        # Process each split
        for split_name, genomes in [
            ("train", train_genomes),
            ("val",   val_genomes),
            ("test",  test_genomes),
        ]:
            all_windows      = []
            padded_genomes   = 0
            remainder_count  = 0
            genes_dropped    = 0   # should always be 0
            pad_tokens_added = 0

            for genome in genomes:
                n = len(genome)
                windows = make_windows(genome, window_size)

                # Verify order is intact (catches bugs early)
                verify_order(genome, windows, window_size)

                # Count what happened
                if n < window_size:
                    padded_genomes   += 1
                    pad_tokens_added += (window_size - n)
                elif n % window_size != 0:
                    remainder_count  += 1

                # Windows appended in genome order — NO shuffling
                all_windows.extend(windows)

            # Write file — order is genome order, biological order preserved
            out_path = exp_dir / f"{split_name}_windows.txt"
            with open(out_path, "w") as fh:
                for w in all_windows:
                    fh.write(" ".join(w) + "\n")

            log.info("  [%s] genomes=%d  windows=%d  padded_genomes=%d"
                     "  remainder_windows=%d  pad_tokens_added=%d",
                     split_name, len(genomes), len(all_windows),
                     padded_genomes, remainder_count, pad_tokens_added)
            log.info("  [%s] → %s", split_name, out_path)

            stats[split_name] = {
                "genomes":           len(genomes),
                "windows":           len(all_windows),
                "padded_genomes":    padded_genomes,
                "remainder_windows": remainder_count,
                "pad_tokens_added":  pad_tokens_added,
                "genes_dropped":     genes_dropped,   # must be 0
            }

        # Save per-experiment stats
        stats_path = exp_dir / "dataset_stats.json"
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2)
        log.info("  Stats → %s", stats_path)

        overall_summary[label] = stats

    # ── Save overall summary ──────────────────────────────────────────────────
    summary_path = base_dir / "phase1_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(overall_summary, fh, indent=2)

    log.info("")
    log.info("=" * 65)
    log.info("PHASE 1 DATASET SUMMARY")
    log.info("=" * 65)
    log.info("  %-10s  %10s  %10s  %10s", "Experiment", "Train", "Val", "Test")
    log.info("  %-10s  %10s  %10s  %10s", "-"*10, "-"*10, "-"*10, "-"*10)
    for label, s in overall_summary.items():
        log.info("  %-10s  %10d  %10d  %10d",
                 label,
                 s["train"]["windows"],
                 s["val"]["windows"],
                 s["test"]["windows"])
    log.info("")
    log.info("  genes_dropped in every experiment should be 0.")
    log.info("  shuffled = False in every experiment.")
    log.info("  Summary → %s", summary_path)
    log.info("✅  Phase 1 datasets ready. Submit with:")
    log.info("    sbatch scripts/slurm/train_phase1.slurm")


if __name__ == "__main__":
    main()