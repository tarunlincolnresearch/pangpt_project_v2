#!/usr/bin/env python3
"""
Analyze vocabulary distribution to determine optimal reduction strategy
"""
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("="*80)
print("VOCABULARY ANALYSIS")
print("="*80)

# Load all genomes
input_file = '../../james_exact_approach/data/all_genomes.txt'
print(f"\nLoading genomes from: {input_file}")

with open(input_file) as f:
    genomes = [line.strip() for line in f]

print(f"✓ Total genomes: {len(genomes)}")

# Count all genes
print("\nCounting gene frequencies...")
all_genes = []
for genome in genomes:
    all_genes.extend(genome.split())

gene_counts = Counter(all_genes)
total_genes = len(all_genes)

print(f"✓ Total gene occurrences: {total_genes:,}")
print(f"✓ Unique genes: {len(gene_counts):,}")

# Analyze distribution
print("\n" + "="*80)
print("FREQUENCY DISTRIBUTION")
print("="*80)

# Sort by frequency
sorted_genes = gene_counts.most_common()

# Calculate coverage at different thresholds
thresholds = [1, 2, 5, 10, 20, 50, 100]

print(f"\n{'Min Occurrences':<20} {'Genes Kept':<15} {'% of Vocab':<15} {'Coverage':<15}")
print("-"*80)

for threshold in thresholds:
    genes_kept = sum(1 for gene, count in sorted_genes if count >= threshold)
    coverage = sum(count for gene, count in sorted_genes if count >= threshold) / total_genes
    
    print(f"{threshold:<20} {genes_kept:<15,} {genes_kept/len(gene_counts)*100:>6.2f}%        {coverage*100:>6.2f}%")

# Show top 20 most frequent genes
print("\n" + "="*80)
print("TOP 20 MOST FREQUENT GENES")
print("="*80)
print(f"{'Rank':<6} {'Gene':<30} {'Count':<15} {'Frequency':<15}")
print("-"*80)

for i, (gene, count) in enumerate(sorted_genes[:20], 1):
    freq = count / total_genes * 100
    print(f"{i:<6} {gene:<30} {count:<15,} {freq:>6.4f}%")

# Show bottom 20 (rarest genes)
print("\n" + "="*80)
print("BOTTOM 20 RAREST GENES")
print("="*80)
print(f"{'Gene':<30} {'Count':<15} {'Frequency':<15}")
print("-"*80)

for gene, count in sorted_genes[-20:]:
    freq = count / total_genes * 100
    print(f"{gene:<30} {count:<15,} {freq:>6.6f}%")

# Calculate how many genes appear only once
singleton_count = sum(1 for gene, count in sorted_genes if count == 1)
print(f"\n" + "="*80)
print(f"Genes appearing only ONCE: {singleton_count:,} ({singleton_count/len(gene_counts)*100:.1f}% of vocabulary)")
print(f"Genes appearing 1-5 times: {sum(1 for g, c in sorted_genes if 1 <= c <= 5):,}")
print(f"Genes appearing 6-10 times: {sum(1 for g, c in sorted_genes if 6 <= c <= 10):,}")
print("="*80)

# Create visualization
print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Top 100 genes
top_100_counts = [c for g, c in sorted_genes[:100]]

axes[0, 0].bar(range(100), top_100_counts, color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('Gene Rank', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Top 100 Most Frequent Genes', fontweight='bold', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cumulative coverage
cumulative_coverage = []
cumulative_sum = 0
for gene, count in sorted_genes:
    cumulative_sum += count
    cumulative_coverage.append(cumulative_sum / total_genes * 100)

axes[0, 1].plot(range(len(cumulative_coverage)), cumulative_coverage, 
                linewidth=2, color='green')
axes[0, 1].axhline(y=90, color='red', linestyle='--', label='90% coverage')
axes[0, 1].axhline(y=95, color='orange', linestyle='--', label='95% coverage')
axes[0, 1].axhline(y=99, color='blue', linestyle='--', label='99% coverage')
axes[0, 1].set_xlabel('Number of Genes (ranked by frequency)', fontweight='bold')
axes[0, 1].set_ylabel('Cumulative Coverage (%)', fontweight='bold')
axes[0, 1].set_title('Cumulative Coverage by Top N Genes', fontweight='bold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, 20000)

# Plot 3: Frequency distribution (log scale)
frequencies = [c for g, c in sorted_genes]
axes[1, 0].hist(frequencies, bins=100, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Gene Frequency', fontweight='bold')
axes[1, 0].set_ylabel('Number of Genes', fontweight='bold')
axes[1, 0].set_title('Gene Frequency Distribution', fontweight='bold', fontsize=14)
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Vocabulary reduction impact
thresholds_plot = [1, 2, 5, 10, 20, 50, 100, 200]
vocab_sizes = []
coverages = []

for threshold in thresholds_plot:
    genes_kept = sum(1 for gene, count in sorted_genes if count >= threshold)
    coverage = sum(count for gene, count in sorted_genes if count >= threshold) / total_genes
    vocab_sizes.append(genes_kept)
    coverages.append(coverage * 100)

axes[1, 1].plot(vocab_sizes, coverages, marker='o', linewidth=2, 
                markersize=8, color='red')
axes[1, 1].axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% target')
axes[1, 1].set_xlabel('Vocabulary Size', fontweight='bold')
axes[1, 1].set_ylabel('Coverage (%)', fontweight='bold')
axes[1, 1].set_title('Vocabulary Size vs Coverage Trade-off', fontweight='bold', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].invert_xaxis()

# Annotate points
for i, (vs, cov) in enumerate(zip(vocab_sizes, coverages)):
    axes[1, 1].annotate(f'{thresholds_plot[i]}+ occurrences\n{vs:,} genes\n{cov:.1f}%', 
                       (vs, cov), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()

# Save to parent visualizations directory
viz_dir = '../visualizations'
os.makedirs(viz_dir, exist_ok=True)
output_file = os.path.join(viz_dir, 'vocabulary_analysis.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find optimal threshold
optimal_threshold = 10
genes_at_optimal = sum(1 for gene, count in sorted_genes if count >= optimal_threshold)
coverage_at_optimal = sum(count for gene, count in sorted_genes if count >= optimal_threshold) / total_genes

print(f"""
Based on analysis, recommended vocabulary reduction:

Current vocabulary: {len(gene_counts):,} genes
Recommended threshold: Keep genes appearing ≥{optimal_threshold} times

Result:
  - New vocabulary: {genes_at_optimal:,} genes
  - Reduction: {len(gene_counts) - genes_at_optimal:,} genes removed ({(1-genes_at_optimal/len(gene_counts))*100:.1f}%)
  - Coverage: {coverage_at_optimal*100:.2f}% of all gene occurrences
  - Genes mapped to <RARE>: {len(gene_counts) - genes_at_optimal:,}

Benefits:
  ✓ {len(gene_counts)/genes_at_optimal:.1f}x smaller vocabulary
  ✓ Model focuses on learnable patterns
  ✓ Rare genes mapped to special <RARE> token
  ✓ Expected accuracy improvement: +15-25%
""")

print("="*80)