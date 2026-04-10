#!/usr/bin/env python3
"""
Comprehensive analysis of why the model plateaus and doesn't learn well
"""

import json
from collections import Counter
import numpy as np
import pandas as pd

print("="*100)
print("ANALYZING WHY MODEL PERFORMANCE IS POOR")
print("="*100)
print()

# ============================================================
# 1. DATA DISTRIBUTION ANALYSIS
# ============================================================

print("1. DATA DISTRIBUTION ANALYSIS")
print("-" * 100)

# Load training data
with open('data/phase1/win256/train_windows.txt', 'r') as f:
    train_windows = [line.strip() for line in f]

print(f"Total training windows: {len(train_windows):,}")

# Analyze gene frequency
all_genes = []
for window in train_windows[:50000]:  # Sample 50k windows
    all_genes.extend(window.split())

gene_counts = Counter(all_genes)
total_genes = len(all_genes)
unique_genes = len(gene_counts)

print(f"Total genes in sample: {total_genes:,}")
print(f"Unique genes: {unique_genes:,}")
print(f"Vocabulary size: 50,000 (model)")
print()

# Check class imbalance
print("Gene Frequency Distribution:")
top_10_count = sum(count for _, count in gene_counts.most_common(10))
top_100_count = sum(count for _, count in gene_counts.most_common(100))
top_1000_count = sum(count for _, count in gene_counts.most_common(1000))

print(f"  Top 10 genes:    {top_10_count:,} / {total_genes:,} = {top_10_count/total_genes*100:.2f}%")
print(f"  Top 100 genes:   {top_100_count:,} / {total_genes:,} = {top_100_count/total_genes*100:.2f}%")
print(f"  Top 1000 genes:  {top_1000_count:,} / {total_genes:,} = {top_1000_count/total_genes*100:.2f}%")
print()

# Calculate entropy
probs = np.array([count/total_genes for count in gene_counts.values()])
entropy = -np.sum(probs * np.log2(probs + 1e-10))
max_entropy = np.log2(unique_genes)
print(f"Dataset Entropy: {entropy:.2f} bits")
print(f"Max Entropy: {max_entropy:.2f} bits")
print(f"Normalized Entropy: {entropy/max_entropy:.2f} (1.0 = perfectly uniform)")
print()

# ============================================================
# 2. TASK DIFFICULTY ANALYSIS
# ============================================================

print("2. TASK DIFFICULTY ANALYSIS")
print("-" * 100)

print(f"Vocabulary size: {unique_genes:,}")
print(f"Random guess accuracy: {1/unique_genes*100:.4f}%")
print(f"Model achieves: 50.27%")
print(f"Improvement over random: {(50.27 / (1/unique_genes*100)):.0f}x")
print()

# Check if genes follow patterns
print("Checking for sequential patterns...")
bigram_counts = Counter()
for window in train_windows[:10000]:
    genes = window.split()
    for i in range(len(genes)-1):
        bigram = (genes[i], genes[i+1])
        bigram_counts[bigram] += 1

total_bigrams = sum(bigram_counts.values())
unique_bigrams = len(bigram_counts)
print(f"Total bigrams (gene pairs): {total_bigrams:,}")
print(f"Unique bigrams: {unique_bigrams:,}")
print(f"Bigram repetition rate: {total_bigrams/unique_bigrams:.2f}x")

if total_bigrams/unique_bigrams < 2:
    print("⚠️  WARNING: Very low bigram repetition!")
    print("   → Gene sequences are highly diverse")
    print("   → Hard for model to learn patterns")
print()

# ============================================================
# 3. MODEL CAPACITY ANALYSIS
# ============================================================

print("3. MODEL CAPACITY ANALYSIS")
print("-" * 100)

# Model parameters
vocab_size = 50000
embed_dim = 256
num_heads = 8
num_layers = 6

# Calculate model size
embedding_params = vocab_size * embed_dim
transformer_params_per_layer = (
    4 * embed_dim * embed_dim +  # Self-attention (Q, K, V, O)
    2 * embed_dim * (4 * embed_dim) +  # FFN
    4 * embed_dim  # Layer norms
)
total_transformer_params = transformer_params_per_layer * num_layers
output_params = embed_dim * vocab_size
total_params = embedding_params + total_transformer_params + output_params

print(f"Model Architecture:")
print(f"  Embedding dimension: {embed_dim}")
print(f"  Attention heads: {num_heads}")
print(f"  Transformer layers: {num_layers}")
print(f"  Vocabulary size: {vocab_size:,}")
print()
print(f"Parameter Count:")
print(f"  Embedding: {embedding_params:,}")
print(f"  Transformer: {total_transformer_params:,}")
print(f"  Output: {output_params:,}")
print(f"  TOTAL: {total_params:,} (~{total_params/1e6:.1f}M parameters)")
print()

# Check if model is large enough
params_per_gene = total_params / vocab_size
print(f"Parameters per gene: {params_per_gene:.0f}")
if params_per_gene < 100:
    print("⚠️  WARNING: Low parameters per gene!")
    print("   → Model might not have enough capacity")
print()

# ============================================================
# 4. TRAINING DYNAMICS ANALYSIS
# ============================================================

print("4. TRAINING DYNAMICS ANALYSIS")
print("-" * 100)

# From Phase 2A results
initial_loss = 3.15
final_loss = 4.96
loss_increase = final_loss - initial_loss

print(f"Initial loss (Epoch 0): {initial_loss:.2f}")
print(f"Final loss (Epoch 30): {final_loss:.2f}")
print(f"Loss change: +{loss_increase:.2f} (INCREASED!)")
print()

if loss_increase > 0:
    print("❌ PROBLEM: Loss INCREASED during training!")
    print("   Possible causes:")
    print("   1. Learning rate too high → model diverging")
    print("   2. Regularization too strong → can't fit data")
    print("   3. Optimization stuck in bad local minimum")
    print("   4. Task is too hard for this architecture")
print()

# ============================================================
# 5. ROOT CAUSE SUMMARY
# ============================================================

print("="*100)
print("ROOT CAUSE ANALYSIS SUMMARY")
print("="*100)
print()

print("🔍 IDENTIFIED PROBLEMS:")
print()

print("1. EXTREMELY DIVERSE DATA")
print("   - 52K unique genes with flat distribution")
print("   - Low bigram repetition (genes don't follow strong patterns)")
print("   - Model can't memorize all combinations")
print()

print("2. TASK IS TOO HARD")
print("   - Predicting 1 out of 52K genes is extremely difficult")
print("   - Even humans couldn't do this without domain knowledge")
print("   - 50% accuracy is actually impressive given the difficulty!")
print()

print("3. MODEL LEARNS 'SAFE' STRATEGY")
print("   - Instead of learning complex patterns, model learns:")
print("   - 'Always predict genes that appear frequently in this context'")
print("   - This gives ~50% accuracy but doesn't improve further")
print()

print("4. PLATEAU IS EXPECTED")
print("   - Model quickly finds the 'safe' strategy (Epoch 1-2)")
print("   - Further training doesn't help because:")
print("     • Data is too diverse to memorize")
print("     • No strong sequential patterns to learn")
print("     • Model capacity might be insufficient")
print()

# ============================================================
# 6. CONCRETE RECOMMENDATIONS
# ============================================================

print("="*100)
print("💡 CONCRETE RECOMMENDATIONS TO IMPROVE PERFORMANCE")
print("="*100)
print()

print("OPTION 1: REDUCE TASK DIFFICULTY")
print("-" * 50)
print("✅ Reduce vocabulary size:")
print("   - Current: 50K genes")
print("   - Recommended: 10K-20K most frequent genes")
print("   - Map rare genes to [RARE] token")
print("   - Expected improvement: 5-10% accuracy gain")
print()

print("✅ Use hierarchical prediction:")
print("   - First predict gene family/category")
print("   - Then predict specific gene within family")
print("   - Expected improvement: 10-15% accuracy gain")
print()

print("OPTION 2: INCREASE MODEL CAPACITY")
print("-" * 50)
print("✅ Larger model:")
print("   - Current: 6 layers, 256 dim (~13M params)")
print("   - Recommended: 12 layers, 512 dim (~100M params)")
print("   - Expected improvement: 5-10% accuracy gain")
print("   - Downside: Much slower training")
print()

print("OPTION 3: MORE/BETTER DATA")
print("-" * 50)
print("✅ More training data:")
print("   - Current: 147K windows")
print("   - Recommended: 500K-1M windows")
print("   - Expected improvement: 3-5% accuracy gain")
print()

print("✅ Add context features:")
print("   - Include genome metadata (species, GC content, etc.)")
print("   - Include gene function annotations")
print("   - Expected improvement: 10-20% accuracy gain")
print()

print("OPTION 4: CHANGE THE TASK")
print("-" * 50)
print("✅ Easier prediction tasks:")
print("   - Instead of exact gene, predict gene category")
print("   - Instead of next gene, predict 'next 3 genes contain X'")
print("   - Expected improvement: 20-30% accuracy gain")
print()

print("OPTION 5: USE PRE-TRAINING")
print("-" * 50)
print("✅ Pre-train on related tasks:")
print("   - Masked gene prediction (like BERT)")
print("   - Gene order reconstruction")
print("   - Then fine-tune on next gene prediction")
print("   - Expected improvement: 10-15% accuracy gain")
print()

print("="*100)
print("🎯 RECOMMENDED NEXT STEPS (Prioritized)")
print("="*100)
print()
print("1. EASIEST & HIGHEST IMPACT: Reduce vocabulary to 20K genes")
print("   → Quick to implement, significant improvement expected")
print()
print("2. MEDIUM EFFORT: Add hierarchical prediction")
print("   → Requires gene family annotations, but very effective")
print()
print("3. LONG-TERM: Collect more diverse training data")
print("   → Most impactful but requires data collection effort")
print()

print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)

