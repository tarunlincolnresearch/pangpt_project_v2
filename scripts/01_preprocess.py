#!/usr/bin/env python3
"""
01_preprocess.py
================
Parse the raw gene-order file, clean all entries, and produce:
  - cleaned_gene_orders.json  (genome → gene list)
  - pangpt_training_sequences.txt  (one genome per line, space-separated)
  - preprocessing_stats.json  (detailed QC report)

Key improvement over original notebook:
  * Counts and logs every "*" entry removed, per genome and globally
  * Flags short / empty genomes
  * Logs malformed lines
  * Writes a machine-readable stats JSON for downstream scripts
"""

import os
import sys
import gzip
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path

# ---------- paths ------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_RAW, DATA_CLEANED_JSON, DATA_SEQUENCES_TXT,
    DATA_STATS_JSON, LOGS_DIR
)

# ---------- logging ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "01_preprocess.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ============================================================================
# HELPERS
# ============================================================================

def open_file(path):
    """Open plain or .gz file transparently."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def parse_gene_order(filepath) -> dict:
    """
    Parse the gene-order file format:
        >genome_id
        gene1,gene2,*,gene3,...

    Returns
    -------
    genomes : dict  {genome_id -> [gene, ...]}
    stats   : dict  detailed counts of everything removed
    """
    genomes = {}
    stats = {
        "total_raw_lines":        0,
        "genome_id_lines":        0,
        "gene_lines":             0,
        "malformed_lines":        0,
        "total_genes_raw":        0,
        "total_star_removed":     0,      # ← key new metric
        "total_empty_removed":    0,
        "total_whitespace_removed": 0,
        "genomes_with_stars":     0,
        "star_counts_per_genome": {},     # genome_id -> n_stars
        "empty_genomes":          [],
        "genome_lengths":         [],
    }

    current_genome = None
    star_in_current = 0

    log.info("Opening input file: %s", filepath)

    with open_file(filepath) as fh:
        for raw_line in fh:
            stats["total_raw_lines"] += 1
            line = raw_line.strip()

            if not line:
                continue

            # ── genome ID line ─────────────────────────────────────────────
            if line.startswith(">"):
                # save star count for previous genome
                if current_genome is not None and star_in_current > 0:
                    stats["star_counts_per_genome"][current_genome] = star_in_current
                    stats["genomes_with_stars"] += 1

                current_genome = line[1:].strip()
                if not current_genome:
                    log.warning("  Malformed genome ID line (empty after >): %r", line)
                    stats["malformed_lines"] += 1
                    current_genome = None
                    continue

                genomes[current_genome] = []
                star_in_current = 0
                stats["genome_id_lines"] += 1

            # ── gene list line ─────────────────────────────────────────────
            else:
                if current_genome is None:
                    log.warning("  Gene line before any genome ID: %r", line[:80])
                    stats["malformed_lines"] += 1
                    continue

                stats["gene_lines"] += 1
                raw_tokens = line.split(",")
                stats["total_genes_raw"] += len(raw_tokens)

                clean_genes = []
                for tok in raw_tokens:
                    tok = tok.strip()
                    if tok == "*":
                        stats["total_star_removed"] += 1
                        star_in_current += 1
                    elif tok == "":
                        stats["total_empty_removed"] += 1
                    elif tok.isspace():
                        stats["total_whitespace_removed"] += 1
                    else:
                        clean_genes.append(tok)

                genomes[current_genome].extend(clean_genes)

    # save final genome's star count
    if current_genome is not None and star_in_current > 0:
        stats["star_counts_per_genome"][current_genome] = star_in_current
        stats["genomes_with_stars"] += 1

    # genome-level stats
    for gid, genes in genomes.items():
        stats["genome_lengths"].append(len(genes))
        if len(genes) == 0:
            stats["empty_genomes"].append(gid)

    return genomes, stats


# ============================================================================
# REPORTING
# ============================================================================

def report_star_stats(stats: dict):
    """Print and log detailed * removal statistics."""
    log.info("=" * 60)
    log.info("STAR (*) ENTRY REMOVAL REPORT")
    log.info("=" * 60)
    log.info("  Total raw tokens parsed      : %d", stats["total_genes_raw"])
    log.info("  Total '*' entries removed    : %d", stats["total_star_removed"])
    log.info("  Total empty tokens removed   : %d", stats["total_empty_removed"])
    log.info("  Genomes containing '*' entries: %d", stats["genomes_with_stars"])

    if stats["total_genes_raw"] > 0:
        star_pct = 100 * stats["total_star_removed"] / stats["total_genes_raw"]
        log.info("  '*' as %% of all raw tokens   : %.3f%%", star_pct)

    # top-20 genomes by star count
    if stats["star_counts_per_genome"]:
        top = sorted(
            stats["star_counts_per_genome"].items(),
            key=lambda x: x[1], reverse=True
        )[:20]
        log.info("")
        log.info("  Top 20 genomes by '*' count:")
        for gid, cnt in top:
            log.info("    %-35s  stars=%d", gid, cnt)

    # distribution of star counts
    counts = list(stats["star_counts_per_genome"].values())
    if counts:
        buckets = Counter()
        for c in counts:
            if c == 1:       buckets["1"] += 1
            elif c <= 5:     buckets["2-5"] += 1
            elif c <= 20:    buckets["6-20"] += 1
            elif c <= 100:   buckets["21-100"] += 1
            else:            buckets["100+"] += 1
        log.info("")
        log.info("  Star-count distribution across affected genomes:")
        for bucket, n in sorted(buckets.items()):
            log.info("    %-10s  %d genomes", bucket, n)

    log.info("=" * 60)


def report_genome_stats(genomes: dict, stats: dict):
    lengths = stats["genome_lengths"]
    if not lengths:
        return
    import statistics
    log.info("GENOME STATISTICS (after cleaning)")
    log.info("  Total genomes              : %d", len(genomes))
    log.info("  Mean gene count            : %.1f", statistics.mean(lengths))
    log.info("  Median gene count          : %.1f", statistics.median(lengths))
    log.info("  Min gene count             : %d",   min(lengths))
    log.info("  Max gene count             : %d",   max(lengths))
    log.info("  Empty genomes (0 genes)    : %d",   len(stats["empty_genomes"]))
    if stats["empty_genomes"]:
        for gid in stats["empty_genomes"][:10]:
            log.warning("    Empty genome: %s", gid)
    log.info("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("▶  Step 1 — Preprocessing")
    log.info("   Input : %s", DATA_RAW)

    if not Path(DATA_RAW).exists():
        log.error("Input file not found: %s", DATA_RAW)
        log.error("Please place gene-order.gz (or gene-order) in %s", DATA_RAW.parent)
        sys.exit(1)

    # ── parse ────────────────────────────────────────────────────────────────
    log.info("Parsing gene-order file …")
    genomes, stats = parse_gene_order(DATA_RAW)

    # ── report ───────────────────────────────────────────────────────────────
    report_star_stats(stats)
    report_genome_stats(genomes, stats)

    # ── remove empty genomes ─────────────────────────────────────────────────
    before = len(genomes)
    genomes = {k: v for k, v in genomes.items() if len(v) >= 10}
    log.info("Removed %d genomes with <10 genes. Remaining: %d",
             before - len(genomes), len(genomes))

    # ── build adjacency pairs (top 10 for report) ────────────────────────────
    log.info("Computing adjacency pairs …")
    adj: defaultdict = defaultdict(int)
    for genes in genomes.values():
        for a, b in zip(genes, genes[1:]):
            adj[(a, b)] += 1
    top_adj = sorted(adj.items(), key=lambda x: x[1], reverse=True)[:10]
    log.info("Total unique adjacency pairs: %d", len(adj))
    log.info("Top 10 gene adjacencies:")
    for (a, b), cnt in top_adj:
        log.info("  (%s, %s) → %d", a, b, cnt)

    # ── write JSON ───────────────────────────────────────────────────────────
    log.info("Writing %s …", DATA_CLEANED_JSON)
    with open(DATA_CLEANED_JSON, "w") as fh:
        json.dump(genomes, fh)

    # ── write space-separated text ───────────────────────────────────────────
    log.info("Writing %s …", DATA_SEQUENCES_TXT)
    total_genes = 0
    with open(DATA_SEQUENCES_TXT, "w") as fh:
        for genes in genomes.values():
            fh.write(" ".join(genes) + "\n")
            total_genes += len(genes)
    log.info("  Genomes written : %d", len(genomes))
    log.info("  Total genes     : %d", total_genes)

    # ── persist stats JSON ───────────────────────────────────────────────────
    stats_out = {k: v for k, v in stats.items() if k != "genome_lengths"}
    stats_out["total_genomes_final"] = len(genomes)
    stats_out["total_genes_final"]   = total_genes
    with open(DATA_STATS_JSON, "w") as fh:
        json.dump(stats_out, fh, indent=2)
    log.info("Stats saved to %s", DATA_STATS_JSON)
    log.info("✅  Preprocessing complete.")


if __name__ == "__main__":
    main()
