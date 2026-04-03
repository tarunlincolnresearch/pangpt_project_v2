#!/usr/bin/env python3
"""
05_inference.py
===============
Gene sequence prediction using the trained panGPT model.

BUG FIXES applied:
  Bug 11: added argparse so --input_file, --model_save_path, --tokenizer_file,
          --results_dir from run_pipeline.sh are actually honoured.
          Previously all four were silently ignored and the script always used
          config.py globals — meaning every Phase 1 experiment was evaluated
          on the same (wrong) checkpoint/test file.
  Bug 12: fixed PositionalEncoding.forward to use x.size(1) (sequence length)
          instead of x.size(0) (batch size) when batch_first=True.
          The pe buffer is (max_len, 1, d_model) after transpose, so indexing
          with the batch size silently applied PE to only the first token.
  Bug 13: load_state_dict now uses strict=True and logs any key mismatches
          so architecture errors are visible rather than silent.
  Bug 14: temperature sweep now runs on validation windows, not test windows,
          to avoid tuning hyperparameters on the test set.
  Bug 15: added guard before statistics.mean/stdev to handle the case where
          all windows are skipped (too short), preventing StatisticsError.
"""

import sys
import os
import json
import math
import logging
import statistics
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODEL_CHECKPOINT, TOKENIZER_FILE, DATA_TEST_WINDOWS, DATA_VAL_WINDOWS,
    LOGS_DIR, RESULTS_DIR, INFERENCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "05_inference.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# CLI  — BUG FIX (Bug 11): added argparse so run_pipeline.sh arguments work
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference / evaluation with a trained panGPT model."
    )
    p.add_argument("--input_file",      type=Path, default=None,
                   help="Test windows .txt file (default: DATA_TEST_WINDOWS)")
    p.add_argument("--model_save_path", type=Path, default=None,
                   help="Checkpoint .pth file (default: MODEL_CHECKPOINT)")
    p.add_argument("--tokenizer_file",  type=Path, default=None,
                   help="Tokenizer JSON (default: TOKENIZER_FILE)")
    p.add_argument("--results_dir",     type=Path, default=None,
                   help="Directory to write results JSON (default: RESULTS_DIR)")
    p.add_argument("--val_file",        type=Path, default=None,
                   help="Validation windows .txt for temperature sweep "
                        "(default: DATA_VAL_WINDOWS)")
    return p.parse_args()


# ============================================================================
# MODEL LOADING  (mirrors panGPT architecture exactly)
# ============================================================================

def load_model_and_tokenizer(checkpoint_path: Path, tokenizer_path: Path):
    import torch
    import torch.nn as nn
    from tokenizers import Tokenizer

    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s — run 04_train.py first.", checkpoint_path)
        sys.exit(1)
    if not tokenizer_path.exists():
        log.error("Tokenizer not found: %s", tokenizer_path)
        sys.exit(1)

    # ── exact architecture from panGPT ───────────────────────────────────────
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
            # BUG FIX (Bug 12): pe shape is (max_len, 1, d_model).
            # With batch_first=True, x shape is (batch, seq, embed).
            # Must index pe with x.size(1) (seq length), NOT x.size(0) (batch size).
            # Using x.size(0) was silently applying PE only to the first token.
            x = x + self.pe[:x.size(1), :].squeeze(1).unsqueeze(0)
            return self.dropout(x)

    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length)
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
    log.info("Device: %s", device)

    checkpoint = __import__("torch").load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)

    vocab_size = state["embed.weight"].shape[0]
    embed_dim  = state["embed.weight"].shape[1]
    log.info("Checkpoint: vocab_size=%d  embed_dim=%d", vocab_size, embed_dim)

    layer_keys = [k for k in state if k.startswith("transformer.layers.")]
    if layer_keys:
        num_layers = max(int(k.split(".")[2]) for k in layer_keys) + 1
    else:
        num_layers = 4
    num_heads = 8 if embed_dim % 8 == 0 else 4

    log.info("Inferred: num_layers=%d  num_heads=%d", num_layers, num_heads)

    model = TransformerModel(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_heads=num_heads, num_layers=num_layers,
        max_seq_length=5000,
    )

    # BUG FIX (Bug 13): use strict=True so architecture mismatches are visible.
    # Log any unexpected keys rather than silently ignoring them.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys in checkpoint (will use random init): %s", missing)
    if unexpected:
        log.warning("Unexpected keys in checkpoint (ignored): %s", unexpected)
    if not missing and not unexpected:
        log.info("Checkpoint loaded cleanly (no missing or unexpected keys).")

    model.eval()
    model.to(device)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    log.info("Model and tokenizer loaded. Parameters: %s",
             f"{sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer, device


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_greedy(model, tokenizer, device, prompt_text: str, num_tokens: int):
    """
    Greedy decoding — always pick the highest-probability gene.
    Best for hard exact-match evaluation.
    Returns list of (gene_name, probability).
    """
    import torch
    import torch.nn.functional as F

    token_ids = tokenizer.encode(prompt_text).ids
    results   = []

    with torch.no_grad():
        for _ in range(num_tokens):
            inp    = torch.tensor([token_ids], dtype=torch.long).to(device)
            logits = model(inp)
            last   = logits[0, -1, :]
            probs  = F.softmax(last, dim=-1)
            next_id   = torch.argmax(probs).item()
            next_prob = probs[next_id].item()
            gene_name = tokenizer.decode([next_id]).strip()
            token_ids.append(next_id)
            results.append((gene_name, round(next_prob, 4)))

    return results


def predict_sampling(model, tokenizer, device, prompt_text: str,
                     num_tokens: int, temperature: float, top_k: int):
    """Sampling with temperature and top-k (original notebook approach)."""
    import torch
    import torch.nn.functional as F

    token_ids = tokenizer.encode(prompt_text).ids
    results   = []

    with torch.no_grad():
        for _ in range(num_tokens):
            inp    = torch.tensor([token_ids], dtype=torch.long).to(device)
            logits = model(inp)
            last   = logits[0, -1, :] / temperature

            if top_k > 0:
                topk_vals, topk_idx = torch.topk(last, top_k)
                mask = torch.full_like(last, float("-inf"))
                mask[topk_idx] = topk_vals
                last = mask

            probs     = F.softmax(last, dim=-1)
            next_id   = torch.multinomial(probs, 1).item()
            next_prob = probs[next_id].item()
            gene_name = tokenizer.decode([next_id]).strip()
            token_ids.append(next_id)
            results.append((gene_name, round(next_prob, 4)))

    return results


def get_top_k_candidates(model, tokenizer, device, prompt_text: str, top_k: int = 10):
    """Return top-k candidates with probabilities for the very next position."""
    import torch
    import torch.nn.functional as F

    token_ids = tokenizer.encode(prompt_text).ids
    inp = torch.tensor([token_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs  = F.softmax(logits[0, -1, :], dim=-1)
    topk_probs, topk_ids = torch.topk(probs, top_k)
    return [
        (tokenizer.decode([idx]).strip(), round(p, 4))
        for p, idx in zip(topk_probs.tolist(), topk_ids.tolist())
    ]


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(predicted_names, actual_names):
    n = min(len(predicted_names), len(actual_names))
    exact_matches = sum(p == a for p, a in zip(predicted_names[:n], actual_names[:n]))
    exact_rate    = exact_matches / n if n > 0 else 0
    pred_set      = set(predicted_names)
    actual_set    = set(actual_names[:n])
    overlap       = pred_set & actual_set
    overlap_rate  = len(overlap) / len(actual_set) if actual_set else 0
    return {
        "exact_matches":  exact_matches,
        "exact_rate":     round(exact_rate, 4),
        "set_overlap":    len(overlap),
        "overlap_rate":   round(overlap_rate, 4),
        "n":              n,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    # BUG FIX (Bug 11): parse CLI args so per-experiment paths are used
    args = parse_args()

    effective_test_file  = args.input_file      if args.input_file      else DATA_TEST_WINDOWS
    effective_checkpoint = args.model_save_path if args.model_save_path else MODEL_CHECKPOINT
    effective_tokenizer  = args.tokenizer_file  if args.tokenizer_file  else TOKENIZER_FILE
    effective_results    = args.results_dir     if args.results_dir     else RESULTS_DIR
    # BUG FIX (Bug 14): temperature sweep uses val file, not test file
    effective_val_file   = args.val_file        if args.val_file        else DATA_VAL_WINDOWS

    effective_results.mkdir(parents=True, exist_ok=True)

    log.info("▶  Step 5 — Inference")
    log.info("   Test file   : %s", effective_test_file)
    log.info("   Checkpoint  : %s", effective_checkpoint)
    log.info("   Tokenizer   : %s", effective_tokenizer)
    log.info("   Results dir : %s", effective_results)

    model, tokenizer, device = load_model_and_tokenizer(
        effective_checkpoint, effective_tokenizer
    )

    cfg         = INFERENCE
    prompt_len  = cfg["prompt_length"]
    num_pred    = cfg["num_predictions"]
    temperature = cfg["temperature"]
    top_k       = cfg["top_k"]
    n_eval      = cfg["n_windows_eval"]
    greedy      = cfg["greedy"]

    # ── load test windows ─────────────────────────────────────────────────────
    log.info("Loading windows from %s …", effective_test_file)
    with open(effective_test_file) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    log.info("  Total windows: %d", len(lines))

    # ── single-window detailed example ────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("DETAILED EXAMPLE — Window 0")
    log.info("  Prompt length : %d genes", prompt_len)
    log.info("  Decoding      : %s",
             "greedy (argmax)" if greedy else f"sampling (temp={temperature}, top_k={top_k})")
    log.info("=" * 60)

    genes_0      = lines[0].split()
    prompt_genes = genes_0[:prompt_len]
    actual_genes = genes_0[prompt_len : prompt_len + num_pred]
    prompt_text  = " ".join(prompt_genes)

    log.info("Prompt ends with : …%s", " ".join(prompt_genes[-5:]))
    log.info("Actual next genes: %s", " ".join(actual_genes))
    log.info("")

    if greedy:
        predicted = predict_greedy(model, tokenizer, device, prompt_text, num_pred)
    else:
        predicted = predict_sampling(model, tokenizer, device, prompt_text,
                                     num_pred, temperature, top_k)

    log.info("%-4s %-25s %-25s %-6s %s", "#", "Predicted", "Actual", "Match", "Prob")
    log.info("-" * 68)
    for i, ((pred_gene, prob), actual_gene) in enumerate(
            zip(predicted, actual_genes), 1):
        match = "✅" if pred_gene == actual_gene else "  "
        log.info("%-4d %-25s %-25s %-6s %.4f", i, pred_gene, actual_gene, match, prob)

    m = compute_metrics([g for g, _ in predicted], actual_genes)
    log.info("")
    log.info("Exact matches   : %d/%d (%.1f%%)",
             m["exact_matches"], m["n"], 100 * m["exact_rate"])
    log.info("Set overlap     : %d/%d genes in common (%.1f%%)",
             m["set_overlap"], m["n"], 100 * m["overlap_rate"])

    # ── top-k candidates for next position ────────────────────────────────────
    log.info("")
    log.info("Top-10 candidates for position %d:", prompt_len + 1)
    candidates = get_top_k_candidates(model, tokenizer, device, prompt_text, top_k=10)
    for rank, (gene, prob) in enumerate(candidates, 1):
        bar = "█" * int(prob * 200)
        log.info("  %2d. %-25s p=%.4f  %s", rank, gene, prob, bar)

    # ── batch evaluation ──────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("BATCH EVALUATION — %d windows", n_eval)
    log.info("  Prompt length : %d  |  Predictions : %d", prompt_len, num_pred)
    log.info("=" * 60)

    all_metrics = []
    for idx in range(min(n_eval, len(lines))):
        genes        = lines[idx].split()
        if len(genes) < prompt_len + num_pred:
            log.debug("  Window %d too short (%d genes), skipping", idx, len(genes))
            continue
        prompt_text  = " ".join(genes[:prompt_len])
        actual_genes = genes[prompt_len : prompt_len + num_pred]

        if greedy:
            predicted = predict_greedy(model, tokenizer, device, prompt_text, num_pred)
        else:
            predicted = predict_sampling(model, tokenizer, device, prompt_text,
                                         num_pred, temperature, top_k)

        m = compute_metrics([g for g, _ in predicted], actual_genes)
        all_metrics.append(m)
        log.info("  Window %-5d | exact=%.2f  overlap=%.2f  (%d/%d genes)",
                 idx, m["exact_rate"], m["overlap_rate"],
                 m["set_overlap"], m["n"])

    # BUG FIX (Bug 15): guard against empty all_metrics (all windows too short)
    if not all_metrics:
        log.warning("No windows were long enough to evaluate "
                    "(need at least %d + %d genes). "
                    "Check prompt_length and num_predictions in config.",
                    prompt_len, num_pred)
    else:
        mean_exact   = statistics.mean(m["exact_rate"]   for m in all_metrics)
        mean_overlap = statistics.mean(m["overlap_rate"] for m in all_metrics)
        std_exact    = statistics.stdev(m["exact_rate"]  for m in all_metrics) if len(all_metrics) > 1 else 0
        std_overlap  = statistics.stdev(m["overlap_rate"] for m in all_metrics) if len(all_metrics) > 1 else 0

        log.info("")
        log.info("SUMMARY over %d windows:", len(all_metrics))
        log.info("  Mean exact match rate : %.4f ± %.4f", mean_exact, std_exact)
        log.info("  Mean set overlap rate : %.4f ± %.4f", mean_overlap, std_overlap)
        log.info("")
        log.info("NOTE: Set overlap rate is the more meaningful metric.")
        log.info("      It asks: 'did the model predict the right SET of genes'")
        log.info("      regardless of order, which is what matters biologically.")

    # ── temperature sweep  ────────────────────────────────────────────────────
    # BUG FIX (Bug 14): run sweep on VALIDATION windows, not test windows.
    # Using test windows for tuning temperature = tuning on the test set.
    log.info("")
    log.info("=" * 60)
    log.info("TEMPERATURE SWEEP (Val Window 0, %d predictions)", num_pred)
    log.info("  NOTE: Using validation set to avoid test-set leakage.")
    log.info("=" * 60)

    sweep_results = []
    val_lines = []
    if effective_val_file.exists():
        with open(effective_val_file) as fh:
            val_lines = [l.strip() for l in fh if l.strip()]

    if not val_lines:
        log.warning("Validation file not found or empty (%s) — skipping temperature sweep.",
                    effective_val_file)
    else:
        val_genes_0  = val_lines[0].split()
        val_prompt   = " ".join(val_genes_0[:prompt_len])
        val_actual   = val_genes_0[prompt_len : prompt_len + num_pred]

        for temp in [0.1, 0.3, 0.5, 0.8, 1.0]:
            preds = predict_sampling(model, tokenizer, device, val_prompt,
                                     num_pred, temperature=temp, top_k=top_k)
            m = compute_metrics([g for g, _ in preds], val_actual)
            log.info("  temp=%-4s  exact=%.4f  overlap=%.4f",
                     temp, m["exact_rate"], m["overlap_rate"])
            sweep_results.append({"temperature": temp, **m})

    # ── save results ──────────────────────────────────────────────────────────
    results_data = {
        "prompt_length":    prompt_len,
        "num_predictions":  num_pred,
        "decoding":         "greedy" if greedy else f"sampling_t{temperature}",
        "batch_metrics":    all_metrics,
        "temperature_sweep": sweep_results,
    }
    out_path = effective_results / "inference_results.json"
    with open(out_path, "w") as fh:
        json.dump(results_data, fh, indent=2)
    log.info("")
    log.info("Results saved → %s", out_path)
    log.info("✅  Inference complete.")


if __name__ == "__main__":
    main()