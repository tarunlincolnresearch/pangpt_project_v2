#!/usr/bin/env python3
"""
04_train.py
===========
Launch panGPT training with the best available configuration.

Priority order for config:
  1. results/best_hyperparam_config.json  (from 03_hyperparam_search.py)
  2. TRAIN dict in config.py              (sensible upgraded defaults)

panGPT code is called as-is via subprocess — nothing in panGPT is modified.

This script also:
  - Copies the existing tokenizer if available (avoids retraining it)
  - Monitors the training log in real-time and prints progress
  - Saves a training summary JSON after completion

BUG FIXES applied:
  Bug 8:  psutil imported lazily inside memory_monitor() so a missing
          install doesn't kill the entire training script at startup
  Bug 9:  removed misleading `global TRAINING_LOG` rebind
  Bug 10: added `from typing import Optional` (was missing, caused NameError)
"""

import sys
import os
import json
import re
import time
import logging
import subprocess
import shutil
import threading
import argparse
from pathlib import Path
from datetime import datetime
# BUG FIX (Bug 10): Optional was used as a return type but never imported
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_TRAIN_WINDOWS, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR,
    PANGPT_SCRIPT, TOKENIZER_FILE, MODEL_CHECKPOINT, TRAIN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "04_train.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

BEST_CONFIG_FILE = RESULTS_DIR / "best_hyperparam_config.json"


# ============================================================================
# CLI ARGUMENT PARSER
# (allows run_pipeline.sh to pass per-experiment paths for Phase 1)
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train panGPT — paths default to config.py values but can "
                    "be overridden via CLI for per-window-size Phase 1 experiments."
    )
    p.add_argument("--input_file",      type=Path, default=None,
                   help="Path to train windows .txt file (default: DATA_TRAIN_WINDOWS)")
    p.add_argument("--model_save_path", type=Path, default=None,
                   help="Where to save the model checkpoint (default: MODEL_CHECKPOINT)")
    p.add_argument("--tokenizer_file",  type=Path, default=None,
                   help="Tokenizer JSON path (default: TOKENIZER_FILE)")
    p.add_argument("--log_dir",         type=Path, default=None,
                   help="Directory for logs (default: LOGS_DIR)")
    p.add_argument("--train_size",      type=float, default=None,
                   help="Fraction of windows for training (overrides config TRAIN value)")
    p.add_argument("--val_size",        type=float, default=None,
                   help="Fraction of windows for validation (overrides config TRAIN value)")
    return p.parse_args()


# ============================================================================
# LOAD CONFIG
# ============================================================================

def load_train_config() -> dict:
    """Load best config from hyperparam search, or fall back to config.py defaults."""
    if BEST_CONFIG_FILE.exists():
        with open(BEST_CONFIG_FILE) as fh:
            cfg = json.load(fh)
        log.info("Loaded config from hyperparameter search: %s", BEST_CONFIG_FILE)
    else:
        cfg = dict(TRAIN)
        log.info("No hyperparam search results found — using config.py defaults.")
    return cfg


# ============================================================================
# BUILD COMMAND
# ============================================================================

def build_command(cfg: dict, input_file=None, model_save_path=None,
                  tokenizer_file=None, log_dir=None) -> list:
    """Build the panGPT command-line invocation."""
    _input       = input_file      if input_file      else DATA_TRAIN_WINDOWS
    _checkpoint  = model_save_path if model_save_path else MODEL_CHECKPOINT
    _tokenizer   = tokenizer_file  if tokenizer_file  else TOKENIZER_FILE
    _log_dir     = log_dir         if log_dir         else LOGS_DIR
    cmd = [
        sys.executable, str(PANGPT_SCRIPT),
        "--input_file",           str(_input),
        "--model_type",           cfg.get("model_type",          "transformer"),
        "--embed_dim",            str(cfg.get("embed_dim",        512)),
        "--num_heads",            str(cfg.get("num_heads",        8)),
        "--num_layers",           str(cfg.get("num_layers",       6)),
        "--max_seq_length",       str(cfg.get("max_seq_length",   512)),
        "--batch_size",           str(cfg.get("batch_size",       4)),
        "--epochs",               str(cfg.get("epochs",           150)),
        "--learning_rate",        str(cfg.get("learning_rate",    0.0001)),
        "--lr_scheduler_factor",  str(cfg.get("lr_scheduler_factor", 0.5)),
        "--lr_patience",          str(cfg.get("lr_patience",      15)),
        "--weight_decay",         str(cfg.get("weight_decay",     0.0001)),
        "--early_stop_patience",  str(cfg.get("early_stop_patience", 30)),
        "--min_delta",            str(cfg.get("min_delta",        0.001)),
        "--max_vocab_size",       str(cfg.get("max_vocab_size",   50000)),
        "--model_dropout_rate",   str(cfg.get("model_dropout_rate", 0.1)),
        "--train_size",           str(cfg.get("train_size",       0.999)),
        "--val_size",             str(cfg.get("val_size",         0.001)),
        "--seed",                 str(cfg.get("seed",             42)),
        "--pe_max_len",           str(cfg.get("pe_max_len",       5000)),
        "--pe_dropout_rate",      str(cfg.get("pe_dropout_rate",  0.1)),
        "--model_save_path",      str(_checkpoint),
        "--log_dir",              str(_log_dir),
    ]

    # reuse existing tokenizer if present
    if _tokenizer.exists():
        cmd += ["--tokenizer_file", str(_tokenizer)]
        log.info("Reusing existing tokenizer: %s", _tokenizer)

    if _checkpoint.exists():
        log.info("Existing checkpoint found — panGPT will resume from: %s", _checkpoint)

    return cmd


# ============================================================================
# PARSE PROGRESS FROM LOG LINE
# ============================================================================

def parse_progress(line: str) -> Optional[str]:
    """Extract a concise progress string from a panGPT log line."""
    m = re.search(
        r"Epoch\s+(\d+).*?Training Loss:\s*([\d.]+).*?Perplexity:\s*([\d.]+)",
        line
    )
    if m:
        return f"  Epoch {m.group(1):>3}  train_loss={m.group(2)}  perplexity={m.group(3)}"

    m = re.search(
        r"Epoch\s+(\d+).*?Validation Loss:\s*([\d.]+).*?Accuracy:\s*([\d.]+)",
        line
    )
    if m:
        return f"           val_loss={m.group(2)}  accuracy={m.group(3)}"

    if "Test Loss" in line:
        m = re.search(r"Perplexity:\s*([\d.]+).*?Accuracy:\s*([\d.]+)", line)
        if m:
            return f"  ── TEST  perplexity={m.group(1)}  accuracy={m.group(2)}"

    return None


# ============================================================================
# MEMORY MONITOR
# BUG FIX (Bug 8): psutil imported lazily here so a missing install does not
# crash the script at import time — training still works without it.
# ============================================================================

def memory_monitor(interval_sec: int = 300, stop_event: threading.Event = None):
    """Log CPU RAM and GPU memory every interval_sec seconds in background."""
    try:
        import psutil
    except ImportError:
        log.warning("psutil not installed — memory monitoring disabled. "
                    "Install with: pip install psutil")
        return

    import subprocess as _sp
    peak_ram_gb = 0.0

    while not stop_event.is_set():
        mem = psutil.virtual_memory()
        used_gb  = (mem.total - mem.available) / 1e9
        total_gb = mem.total / 1e9
        peak_ram_gb = max(peak_ram_gb, used_gb)

        try:
            result = _sp.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            gpu_line = result.stdout.strip().split("\n")[0]
            gpu_used_mb, gpu_total_mb = [int(x.strip()) for x in gpu_line.split(",")]
            gpu_str = f"GPU {gpu_used_mb} MB / {gpu_total_mb} MB"
        except Exception:
            gpu_str = "GPU N/A"

        log.info(
            "📊 MEMORY  RAM: %.1f GB / %.1f GB (peak %.1f GB)  |  %s",
            used_gb, total_gb, peak_ram_gb, gpu_str
        )
        stop_event.wait(interval_sec)

    log.info("📊 MEMORY FINAL PEAK  RAM: %.1f GB", peak_ram_gb)


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Resolve effective paths: CLI overrides take precedence over config.py defaults
    effective_input       = args.input_file      if args.input_file      else DATA_TRAIN_WINDOWS
    effective_checkpoint  = args.model_save_path if args.model_save_path else MODEL_CHECKPOINT
    effective_tokenizer   = args.tokenizer_file  if args.tokenizer_file  else TOKENIZER_FILE
    effective_log_dir     = args.log_dir         if args.log_dir         else LOGS_DIR

    # Ensure per-experiment log/checkpoint dirs exist
    effective_log_dir.mkdir(parents=True, exist_ok=True)
    effective_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # BUG FIX (Bug 9): removed misleading `global TRAINING_LOG` rebind.
    # We just use the local variable directly everywhere below.
    training_log = effective_log_dir / "training.log"

    log.info("▶  Step 4 — Training")
    log.info("   panGPT : %s", PANGPT_SCRIPT)
    log.info("   Data   : %s", effective_input)

    if not PANGPT_SCRIPT.exists():
        log.error("panGPT script not found. Run: git clone https://github.com/mol-evol/panGPT ~/panGPT")
        sys.exit(1)
    if not effective_input.exists():
        log.error("Windows file not found: %s. Run steps 01 and 02 first.", effective_input)
        sys.exit(1)

    cfg = load_train_config()
    # Apply CLI train/val size overrides
    if args.train_size is not None:
        cfg["train_size"] = args.train_size
    if args.val_size is not None:
        cfg["val_size"] = args.val_size

    log.info("")
    log.info("Training configuration:")
    for k, v in cfg.items():
        log.info("  %-28s = %s", k, v)
    log.info("")

    cmd = build_command(cfg, effective_input, effective_checkpoint,
                        effective_tokenizer, effective_log_dir)
    log.info("Command:\n  %s\n", " \\\n    ".join(cmd))

    start_time = datetime.now()
    log.info("Training started at %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    # Start memory monitor (logs every 5 minutes in background)
    _stop_monitor = threading.Event()
    _monitor_thread = threading.Thread(
        target=memory_monitor,
        kwargs={"interval_sec": 300, "stop_event": _stop_monitor},
        daemon=True,
    )
    _monitor_thread.start()
    log.info("📊 Memory monitor started (logging every 5 min)")

    # ── run panGPT ────────────────────────────────────────────────────────────
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    full_output = []
    with open(training_log, "w") as log_fh:
        for line in proc.stdout:
            full_output.append(line)
            log_fh.write(line)
            log_fh.flush()

            progress = parse_progress(line.strip())
            if progress:
                log.info(progress)
            elif any(kw in line for kw in
                     ["Parameters:", "Dataset loaded", "device =",
                      "EarlyStopping", "Saving model", "Starting training",
                      "Continuing training", "Test Loss"]):
                log.info("  %s", line.strip())

    proc.wait()

    # Stop memory monitor
    _stop_monitor.set()
    _monitor_thread.join(timeout=10)

    end_time = datetime.now()
    elapsed  = end_time - start_time

    log.info("=" * 60)
    log.info("Training finished. Return code: %d", proc.returncode)
    log.info("Duration: %s", str(elapsed).split(".")[0])
    log.info("Full log saved: %s", training_log)

    # ── parse final test metrics ──────────────────────────────────────────────
    combined = "".join(full_output)
    test_match = re.search(
        r"Test Loss:\s*([\d.]+),\s*Perplexity:\s*([\d.]+),\s*Accuracy:\s*([\d.]+)"
        r".*?F1:\s*([\d.]+).*?Kappa:\s*([\d.]+)",
        combined
    )

    summary = {
        "config":       cfg,
        "start_time":   start_time.isoformat(),
        "end_time":     end_time.isoformat(),
        "elapsed_sec":  elapsed.total_seconds(),
        "return_code":  proc.returncode,
    }

    if test_match:
        summary["test_loss"]        = float(test_match.group(1))
        summary["test_perplexity"]  = float(test_match.group(2))
        summary["test_accuracy"]    = float(test_match.group(3))
        summary["test_f1"]          = float(test_match.group(4))
        summary["test_kappa"]       = float(test_match.group(5))
        log.info("")
        log.info("FINAL TEST RESULTS:")
        log.info("  Perplexity : %.4f", summary["test_perplexity"])
        log.info("  Accuracy   : %.4f", summary["test_accuracy"])
        log.info("  F1 Score   : %.4f", summary["test_f1"])
        log.info("  Kappa      : %.4f", summary["test_kappa"])

    summary_path = effective_checkpoint.parent / "training_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary saved → %s", summary_path)

    if proc.returncode != 0:
        log.error("panGPT exited with error. Check %s for details.", training_log)
        sys.exit(1)

    log.info("✅  Training complete. Checkpoint: %s", effective_checkpoint)


if __name__ == "__main__":
    main()