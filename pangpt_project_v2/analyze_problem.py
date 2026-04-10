#!/usr/bin/env python3
"""
Diagnose why the model isn't learning
"""
import json
from collections import Counter

# Load training data
with open('data/phase1/win256/train_windows.txt', 'r') as f:
    train_data = [line.strip() for line in f]

print("=== DATA ANALYSIS ===\n")

# 1. Check vocabulary distribution
all_genes = []
for window in train_data[:10000]:  # Sample first 10k
    all_genes.extend(window.split())

gene_counts = Counter(all_genes)
print(f"Total genes in sample: {len(all_genes)}")
print(f"Unique genes: {len(gene_counts)}")
print(f"\nTop 20 most frequent genes:")
for gene, count in gene_counts.most_common(20):
    pct = (count / len(all_genes)) * 100
    print(f"  {gene}: {count} ({pct:.2f}%)")

# 2. Check class imbalance
top_10_pct = sum(count for _, count in gene_counts.most_common(10)) / len(all_genes) * 100
print(f"\nTop 10 genes account for: {top_10_pct:.1f}% of all occurrences")

# 3. Check if task is too hard
print(f"\nIf top 10 genes = {top_10_pct:.1f}%, then:")
print(f"  - Random guessing from top 10: ~{top_10_pct/10:.1f}% accuracy")
print(f"  - Our model gets: 50.27% accuracy")
print(f"  - This suggests model is just predicting common genes!")

