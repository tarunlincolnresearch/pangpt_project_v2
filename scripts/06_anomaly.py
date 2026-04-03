#!/usr/bin/env python3
"""
06_anomaly.py
=============
Perplexity-based anomaly detection for engineered genomic elements.

How it works
------------
1. Build a baseline: compute perplexity of N normal genomes → get mean/std
2. Set a threshold: mean + 3σ (or 99th percentile)
3. Score a genome: compute its perplexity under the trained model
4. Flag it: if perplexity > threshold → potentially engineered

Synthetic benchmark
-------------------
To test that detection works, we create synthetic anomalies by:
  - Insertion: insert random/foreign gene tokens at random positions
  - Deletion:  remove a contiguous block of genes
  - Substitution: replace a block with random tokens

Expected result: anomalous windows score significantly higher perplexity.

BUG FIXES applied:
  Bug 16: fixed PositionalEncoding.forward to use x.size(1) (sequence length)
          instead of x.size(0) (batch size) when batch_first=True — same fix
          as 05_inference.py.
  Bug 17: substitution modification now uses cfg["substitution_lengths"]
          instead of silently falling back to deletion_lengths.
          substitution_lengths is defined in config.py (added in that fix).
"""

import sys
import json
import math
import random
import logging
import statistics
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODEL_CHECKPOINT, TOKENIZER_FILE, DATA_TEST_WINDOWS,
    LOGS_DIR, RESULTS_DIR, ANOMALY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "06_anomaly.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# CLI  — added for consistency with rest of pipeline
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Perplexity-based anomaly detection with a trained panGPT model."
    )
    p.add_argument("--model_save_path", type=Path, default=None,
                   help="Checkpoint .pth file (default: MODEL_CHECKPOINT)")
    p.add_argument("--tokenizer_file",  type=Path, default=None,
                   help="Tokenizer JSON (default: TOKENIZER_FILE)")
    p.add_argument("--results_dir",     type=Path, default=None,
                   help="Directory to write results JSON (default: RESULTS_DIR)")
    p.add_argument("--test_file",       type=Path, default=None,
                   help="Test windows .txt file (default: DATA_TEST_WINDOWS)")
    return p.parse_args()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(checkpoint_path: Path, tokenizer_path: Path):
    import torch
    import torch.nn as nn
    from tokenizers import Tokenizer

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer("pe", pe)

        def forward(self, x):
            # BUG FIX (Bug 16): pe shape is (max_len, 1, d_model).
            # With batch_first=True, x shape is (batch, seq, embed).
            # Must index with x.size(1) (seq length), NOT x.size(0) (batch size).
            x = x + self.pe[:x.size(1), :].squeeze(1).unsqueeze(0)
            return self.dropout(x)

    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers)
            self.out = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            x = self.embed(x)
            x = self.pos_encoding(x)
            x = self.transformer(x)
            return self.out(x)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    checkpoint = __import__("torch").load(checkpoint_path, map_location=device)
    state      = checkpoint.get("model_state_dict", checkpoint)
    vocab_size = state["embed.weight"].shape[0]
    embed_dim  = state["embed.weight"].shape[1]
    layer_keys = [k for k in state if k.startswith("transformer.layers.")]
    num_layers = (max(int(k.split(".")[2]) for k in layer_keys) + 1) if layer_keys else 4
    num_heads  = 8 if embed_dim % 8 == 0 else 4

    model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, 5000)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys in checkpoint (random init): %s", missing)
    if unexpected:
        log.warning("Unexpected keys in checkpoint (ignored): %s", unexpected)
    if not missing and not unexpected:
        log.info("Checkpoint loaded cleanly.")

    model.eval()
    model.to(device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    log.info("Model loaded on %s | vocab=%d embed=%d layers=%d heads=%d",
             device, vocab_size, embed_dim, num_layers, num_heads)
    return model, tokenizer, device, vocab_size


# ============================================================================
# PERPLEXITY SCORING
# ============================================================================

def compute_perplexity(model, tokenizer, device, gene_text: str,
                       window_size: int = 512) -> float:
    """
    Compute next-token cross-entropy loss over a gene sequence,
    then return exp(loss) as perplexity.
    Uses a sliding window if sequence is longer than window_size.
    """
    import torch
    import torch.nn.functional as F

    token_ids = tokenizer.encode(gene_text).ids
    if len(token_ids) < 2:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, window_size // 2):
            chunk = token_ids[start : start + window_size]
            if len(chunk) < 2:
                break
            inp    = torch.tensor([chunk[:-1]], dtype=torch.long).to(device)
            target = torch.tensor([chunk[1:]],  dtype=torch.long).to(device)
            logits = model(inp)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                reduction="sum"
            )
            total_loss   += loss.item()
            total_tokens += target.numel()

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ============================================================================
# SYNTHETIC ANOMALY GENERATION
# ============================================================================

def make_insertion(genes: list, n_insert: int, vocab_size: int,
                   tokenizer, position: Optional[int] = None) -> list:
    """Insert n random gene tokens at a random (or given) position."""
    genes = genes[:]
    pos = position if position is not None else random.randint(0, len(genes))
    foreign = [f"FOREIGN_gene_{random.randint(100000, 999999)}" for _ in range(n_insert)]
    return genes[:pos] + foreign + genes[pos:]


def make_deletion(genes: list, n_delete: int,
                  position: Optional[int] = None) -> list:
    """Delete n contiguous genes from a random (or given) position."""
    genes = genes[:]
    if len(genes) <= n_delete:
        return genes
    pos = position if position is not None else random.randint(0, len(genes) - n_delete)
    return genes[:pos] + genes[pos + n_delete:]


def make_substitution(genes: list, n_sub: int,
                      position: Optional[int] = None) -> list:
    """Replace n genes with foreign sequences at a random (or given) position."""
    genes = genes[:]
    if len(genes) <= n_sub:
        return genes
    pos = position if position is not None else random.randint(0, len(genes) - n_sub)
    foreign = [f"ENGINEERED_{random.randint(100000, 999999)}" for _ in range(n_sub)]
    return genes[:pos] + foreign + genes[pos + n_sub:]


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    effective_checkpoint = args.model_save_path if args.model_save_path else MODEL_CHECKPOINT
    effective_tokenizer  = args.tokenizer_file  if args.tokenizer_file  else TOKENIZER_FILE
    effective_results    = args.results_dir     if args.results_dir     else RESULTS_DIR
    effective_test_file  = args.test_file       if args.test_file       else DATA_TEST_WINDOWS

    effective_results.mkdir(parents=True, exist_ok=True)

    log.info("▶  Step 6 — Anomaly Detection")
    log.info("   Checkpoint  : %s", effective_checkpoint)
    log.info("   Tokenizer   : %s", effective_tokenizer)
    log.info("   Test file   : %s", effective_test_file)
    log.info("   Results dir : %s", effective_results)

    if not effective_checkpoint.exists():
        log.error("Model checkpoint not found. Run 04_train.py first.")
        sys.exit(1)

    model, tokenizer, device, vocab_size = load_model_and_tokenizer(
        effective_checkpoint, effective_tokenizer
    )
    cfg = ANOMALY
    random.seed(42)

    # ── load windows ──────────────────────────────────────────────────────────
    log.info("Loading windows from %s …", effective_test_file)
    with open(effective_test_file) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    log.info("  Total windows: %d", len(lines))

    # ── build baseline (normal genomes) ───────────────────────────────────────
    n_baseline = min(cfg["baseline_n_genomes"], len(lines))
    log.info("")
    log.info("Building baseline from %d normal windows …", n_baseline)

    baseline_perps = []
    for i in range(n_baseline):
        p = compute_perplexity(model, tokenizer, device, lines[i])
        baseline_perps.append(p)
        if (i + 1) % 20 == 0:
            log.info("  Scored %d/%d  (current mean=%.3f)",
                     i + 1, n_baseline, statistics.mean(baseline_perps))

    baseline_mean = statistics.mean(baseline_perps)
    baseline_std  = statistics.stdev(baseline_perps)
    p99 = sorted(baseline_perps)[int(0.99 * len(baseline_perps))]
    threshold_sigma  = baseline_mean + cfg["threshold_sigma"] * baseline_std
    threshold = min(threshold_sigma, p99)

    log.info("")
    log.info("BASELINE STATISTICS:")
    log.info("  Mean perplexity  : %.4f", baseline_mean)
    log.info("  Std dev          : %.4f", baseline_std)
    log.info("  99th percentile  : %.4f", p99)
    log.info("  Threshold (3σ)   : %.4f", threshold_sigma)
    log.info("  Threshold (used) : %.4f  (min of 3σ and p99)", threshold)

    # ── synthetic anomaly benchmark ───────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SYNTHETIC ANOMALY BENCHMARK")
    log.info("=" * 60)

    results = {
        "baseline_mean":    baseline_mean,
        "baseline_std":     baseline_std,
        "threshold":        threshold,
        "modification_results": {}
    }

    test_windows = lines[n_baseline : n_baseline + cfg["n_synthetic_anomalies"]]

    for mod_type in cfg["modification_types"]:
        # BUG FIX (Bug 17): substitution now uses its own dedicated length list
        # from config instead of silently reusing deletion_lengths.
        if mod_type == "insertion":
            sizes = cfg["insertion_lengths"]
        elif mod_type == "substitution":
            sizes = cfg.get("substitution_lengths", cfg["deletion_lengths"])
        else:  # deletion
            sizes = cfg["deletion_lengths"]

        mod_results = []

        log.info("")
        log.info("Modification: %s", mod_type.upper())

        for size in sizes:
            true_positives  = 0
            false_negatives = 0
            anomaly_perps   = []

            for window_text in test_windows:
                genes = window_text.split()
                if mod_type == "insertion":
                    modified = make_insertion(genes, size, vocab_size, tokenizer)
                elif mod_type == "deletion":
                    modified = make_deletion(genes, size)
                else:  # substitution
                    modified = make_substitution(genes, size)

                modified_text = " ".join(modified)
                perp = compute_perplexity(model, tokenizer, device, modified_text)
                anomaly_perps.append(perp)

                if perp > threshold:
                    true_positives += 1
                else:
                    false_negatives += 1

            detection_rate    = true_positives / len(test_windows) if test_windows else 0
            mean_anomaly_perp = statistics.mean(anomaly_perps)
            perp_increase_pct = 100 * (mean_anomaly_perp - baseline_mean) / baseline_mean

            log.info("  size=%-3d | detection_rate=%.2f%%  mean_perp=%.3f  (+%.1f%% over baseline)",
                     size, 100 * detection_rate, mean_anomaly_perp, perp_increase_pct)

            mod_results.append({
                "modification_size":     size,
                "detection_rate":        round(detection_rate, 4),
                "true_positives":        true_positives,
                "false_negatives":       false_negatives,
                "mean_anomaly_perp":     round(mean_anomaly_perp, 4),
                "perp_increase_pct":     round(perp_increase_pct, 2),
            })

        results["modification_results"][mod_type] = mod_results

    # ── summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("DETECTION SUMMARY")
    log.info("=" * 60)
    for mod_type, mod_res in results["modification_results"].items():
        log.info("%s:", mod_type.upper())
        for r in mod_res:
            log.info("  size=%-3d  detected=%d%%  Δperplexity=+%.1f%%",
                     r["modification_size"],
                     int(100 * r["detection_rate"]),
                     r["perp_increase_pct"])

    out_path = effective_results / "anomaly_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("")
    log.info("Results saved → %s", out_path)
    log.info("✅  Anomaly detection complete.")


if __name__ == "__main__":
    main()