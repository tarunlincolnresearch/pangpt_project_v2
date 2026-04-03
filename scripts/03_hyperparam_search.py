#!/usr/bin/env python3
"""
03_hyperparam_search.py
=======================
Grid search over panGPT training hyperparameters.
Runs short training trials (epochs_per_trial epochs each) and ranks
configurations by final validation perplexity.

After the search completes, it writes the best config back into config.py
(as a JSON sidecar) so 04_train.py automatically picks it up.

Usage
-----
    python scripts/03_hyperparam_search.py

On GPU cluster:
    sbatch scripts/slurm/hyperparam.slurm

BUG FIXES applied:
  Bug 4: added argparse so --input_file from run_pipeline.sh is honoured
  Bug 5: replaced float | None with Optional[float] (Python 3.9 compat)
  Bug 6: removed silent num_heads override that defeated the grid search
  Bug 7: added trial dir cleanup on failure
"""

import sys
import os
import json
import re
import logging
import itertools
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
# BUG FIX (Bug 5): use Optional from typing for Python 3.9 compatibility
from typing import Optional
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_TRAIN_WINDOWS, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR,
    PANGPT_SCRIPT, HYPERPARAM_SEARCH, TOKENIZER_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "03_hyperparam_search.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

BEST_CONFIG_FILE = RESULTS_DIR / "best_hyperparam_config.json"


# ============================================================================
# CLI  — BUG FIX (Bug 4): added argparse so run_pipeline.sh --input_file works
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter search for panGPT. "
                    "--input_file overrides DATA_TRAIN_WINDOWS from config."
    )
    p.add_argument("--input_file", type=Path, default=None,
                   help="Path to train windows .txt file (default: DATA_TRAIN_WINDOWS)")
    return p.parse_args()


# ============================================================================
# PARSE TRAINING LOG
# ============================================================================

# BUG FIX (Bug 5): Optional[float] instead of float | None
def parse_val_perplexity(log_text: str) -> Optional[float]:
    """Extract the lowest validation perplexity from a panGPT training log."""
    pattern = r"Validation Loss:.*?Perplexity:\s*([\d.]+)"
    matches = re.findall(pattern, log_text)
    if not matches:
        return None
    return min(float(m) for m in matches)


def parse_final_accuracy(log_text: str) -> Optional[float]:
    pattern = r"Validation Loss:.*?Accuracy:\s*([\d.]+)"
    matches = re.findall(pattern, log_text)
    if not matches:
        return None
    return float(matches[-1])


# ============================================================================
# RUN ONE TRIAL
# ============================================================================

def run_trial(trial_id: int, params: dict, trial_dir: Path,
              input_file: Path) -> dict:
    """
    Run panGPT for a short number of epochs and return metrics.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_path = trial_dir / "checkpoint.pth"
    log_dir    = trial_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(PANGPT_SCRIPT),
        "--input_file",          str(input_file),
        "--model_type",          "transformer",
        "--embed_dim",           str(params["embed_dim"]),
        "--num_heads",           str(params["num_heads"]),
        "--num_layers",          str(params["num_layers"]),
        "--max_seq_length",      str(params["max_seq_length"]),
        "--batch_size",          str(params["batch_size"]),
        "--epochs",              str(params["epochs_per_trial"]),
        "--learning_rate",       str(params["learning_rate"]),
        "--max_vocab_size",      str(params["max_vocab_size"]),
        "--model_dropout_rate",  str(params["model_dropout_rate"]),
        "--model_save_path",     str(model_path),
        "--log_dir",             str(log_dir),
        "--train_size",          "0.999",
        "--val_size",            "0.001",
        "--seed",                "42",
    ]

    # copy tokenizer if it exists (panGPT will reuse it)
    if TOKENIZER_FILE.exists():
        shutil.copy(TOKENIZER_FILE, trial_dir / "pangenome_gpt_tokenizer.json")
        cmd += ["--tokenizer_file", str(trial_dir / "pangenome_gpt_tokenizer.json")]

    log.info("  Trial %02d | embed=%d layers=%d heads=%d lr=%s dropout=%s batch=%d",
             trial_id,
             params["embed_dim"], params["num_layers"], params["num_heads"],
             params["learning_rate"], params["model_dropout_rate"],
             params["batch_size"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=3600,  # 1-hour per trial safety cap
        )
        combined_output = result.stdout + result.stderr

        val_perp = parse_val_perplexity(combined_output)
        accuracy = parse_final_accuracy(combined_output)

        if val_perp is None:
            log.warning("    Could not parse perplexity — trial may have failed.")
            log.debug("    stdout: %s", result.stdout[-500:])
            log.debug("    stderr: %s", result.stderr[-500:])
            val_perp = float("inf")

        log.info("    → val_perplexity=%.4f  accuracy=%s",
                 val_perp, f"{accuracy:.4f}" if accuracy else "n/a")

        # save trial log
        with open(trial_dir / "trial.log", "w") as fh:
            fh.write(combined_output)

        return {
            "trial_id":        trial_id,
            "params":          params,
            "val_perplexity":  val_perp,
            "accuracy":        accuracy,
            "return_code":     result.returncode,
        }

    except subprocess.TimeoutExpired:
        log.error("    Trial %02d timed out.", trial_id)
        # BUG FIX (Bug 7): clean up partial directory on failure
        shutil.rmtree(trial_dir, ignore_errors=True)
        return {"trial_id": trial_id, "params": params,
                "val_perplexity": float("inf"), "accuracy": None}
    except Exception as exc:
        log.error("    Trial %02d failed: %s", trial_id, exc)
        # BUG FIX (Bug 7): clean up partial directory on failure
        shutil.rmtree(trial_dir, ignore_errors=True)
        return {"trial_id": trial_id, "params": params,
                "val_perplexity": float("inf"), "accuracy": None}


# ============================================================================
# MAIN
# ============================================================================

def main():
    # BUG FIX (Bug 4): parse args so --input_file is respected
    args = parse_args()
    effective_input = args.input_file if args.input_file else DATA_TRAIN_WINDOWS

    log.info("▶  Step 3 — Hyperparameter Search")
    log.info("   panGPT script : %s", PANGPT_SCRIPT)
    log.info("   Input file    : %s", effective_input)

    if not PANGPT_SCRIPT.exists():
        log.error("panGPT script not found at %s", PANGPT_SCRIPT)
        log.error("Run:  git clone https://github.com/mol-evol/panGPT ~/panGPT")
        sys.exit(1)

    if not effective_input.exists():
        log.error("Windows file not found: %s. Run 01_preprocess.py and 02_windowing.py first.",
                  effective_input)
        sys.exit(1)

    cfg = HYPERPARAM_SEARCH
    epochs_per_trial = cfg["epochs_per_trial"]
    n_trials         = cfg["n_trials"]

    # ── build search grid ─────────────────────────────────────────────────────
    grid_keys = ["embed_dim", "num_layers", "num_heads",
                 "learning_rate", "model_dropout_rate", "batch_size"]
    grid_vals = [cfg[k] for k in grid_keys]
    all_combos = list(itertools.product(*grid_vals))

    log.info("Search space: %d combinations, sampling %d trials",
             len(all_combos), n_trials)

    import random
    random.seed(42)
    sampled = random.sample(all_combos, min(n_trials, len(all_combos)))

    search_dir = LOGS_DIR / f"hyperparam_search_{datetime.now():%Y%m%d_%H%M%S}"
    search_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for trial_id, combo in enumerate(sampled):
        params = dict(zip(grid_keys, combo))
        params["epochs_per_trial"] = epochs_per_trial
        params["max_seq_length"]   = cfg["max_seq_length"]
        params["max_vocab_size"]   = cfg["max_vocab_size"]
        # BUG FIX (Bug 6): removed the silent override that always forced
        # num_heads=8 regardless of what was sampled from the grid.
        # num_heads is now used exactly as sampled.

        # ensure embed_dim divisible by num_heads
        if params["embed_dim"] % params["num_heads"] != 0:
            log.info("  Skipping: embed_dim=%d not divisible by num_heads=%d",
                     params["embed_dim"], params["num_heads"])
            continue

        trial_dir = search_dir / f"trial_{trial_id:02d}"
        result    = run_trial(trial_id, params, trial_dir, effective_input)
        results.append(result)

    # ── rank ──────────────────────────────────────────────────────────────────
    valid = [r for r in results if r["val_perplexity"] < float("inf")]
    if not valid:
        log.error("All trials failed — cannot determine best config.")
        sys.exit(1)

    valid.sort(key=lambda r: r["val_perplexity"])

    log.info("")
    log.info("=" * 60)
    log.info("SEARCH RESULTS (ranked by validation perplexity)")
    log.info("=" * 60)
    for rank, r in enumerate(valid[:10], 1):
        p = r["params"]
        log.info(
            "  #%02d  perp=%.4f  acc=%s  embed=%d layers=%d lr=%s dropout=%s batch=%d",
            rank, r["val_perplexity"],
            f"{r['accuracy']:.4f}" if r["accuracy"] else "n/a",
            p["embed_dim"], p["num_layers"], p["learning_rate"],
            p["model_dropout_rate"], p["batch_size"]
        )

    best = valid[0]
    log.info("")
    log.info("BEST CONFIG:")
    for k, v in best["params"].items():
        log.info("  %-25s = %s", k, v)

    # ── save best config ──────────────────────────────────────────────────────
    best_train_config = {
        "embed_dim":           best["params"]["embed_dim"],
        "num_heads":           best["params"]["num_heads"],
        "num_layers":          best["params"]["num_layers"],
        "learning_rate":       best["params"]["learning_rate"],
        "model_dropout_rate":  best["params"]["model_dropout_rate"],
        "batch_size":          best["params"]["batch_size"],
        "max_seq_length":      cfg["max_seq_length"],
        "max_vocab_size":      cfg["max_vocab_size"],
        # keep full-run training settings
        "epochs":              150,
        "model_type":          "transformer",
        "lr_scheduler_factor": 0.5,
        "lr_patience":         15,
        "weight_decay":        0.0001,
        "early_stop_patience": 30,
        "min_delta":           0.001,
        "train_size":          0.999,
        "val_size":            0.001,
        "seed":                42,
        "pe_max_len":          5000,
        "pe_dropout_rate":     0.1,
    }

    with open(BEST_CONFIG_FILE, "w") as fh:
        json.dump(best_train_config, fh, indent=2)
    log.info("Best config saved → %s", BEST_CONFIG_FILE)
    log.info("04_train.py will automatically use this config.")

    # save all results
    with open(RESULTS_DIR / "hyperparam_all_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    log.info("✅  Hyperparameter search complete.")


if __name__ == "__main__":
    main()