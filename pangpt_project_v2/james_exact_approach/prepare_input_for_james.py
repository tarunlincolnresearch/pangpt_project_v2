#!/usr/bin/env python3
"""
Prepare input file for James's panGPT.py
- One genome per line
- Space-separated genes
- No windowing, no pre-splitting
- Let James's code handle everything
"""

import json
import numpy as np

print("="*100)
print("PREPARING INPUT FOR JAMES'S CODE")
print("="*100)
print()

# Load original cleaned gene orders
print("1. Loading original genome data...")
input_file = '../data/cleaned_gene_orders.json'

try:
    with open(input_file, 'r') as f:
        gene_orders = json.load(f)
    print(f"   ✅ Loaded {len(gene_orders):,} genomes")
except FileNotFoundError:
    print(f"   ❌ File not found: {input_file}")
    print(f"   Looking for alternative locations...")
    
    # Try alternative path
    alt_path = '../../data/cleaned_gene_orders.json'
    try:
        with open(alt_path, 'r') as f:
            gene_orders = json.load(f)
        print(f"   ✅ Loaded {len(gene_orders):,} genomes from {alt_path}")
    except:
        print(f"   ❌ Could not find cleaned_gene_orders.json")
        print(f"   Please check the path!")
        exit(1)

print()

# Analyze genome lengths
print("2. Analyzing genome lengths...")
genome_lengths = [len(genes) for genes in gene_orders.values()]
min_length = min(genome_lengths)
max_length = max(genome_lengths)
avg_length = np.mean(genome_lengths)
median_length = np.median(genome_lengths)

print(f"   Min length: {min_length:,} genes")
print(f"   Max length: {max_length:,} genes")
print(f"   Average: {avg_length:,.0f} genes")
print(f"   Median: {median_length:,.0f} genes")
print()

# Show distribution
print("3. Length distribution:")
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    val = int(np.percentile(genome_lengths, p))
    count_over = sum(1 for l in genome_lengths if l > val)
    print(f"   {p}th percentile: {val:,} genes ({count_over} genomes longer)")

print()

# Recommend max_seq_length
recommended_max = int(np.percentile(genome_lengths, 95))
print(f"4. Recommended max_seq_length: {recommended_max:,} (95th percentile)")
print(f"   - Will truncate {sum(1 for l in genome_lengths if l > recommended_max)} genomes")
print(f"   - Will pad {sum(1 for l in genome_lengths if l < recommended_max)} genomes")
print()

# Create single input file (all genomes)
print("5. Creating input file for James's code...")
output_file = 'data/all_genomes.txt'

with open(output_file, 'w') as f:
    for genome_id, genes in gene_orders.items():
        # Write one genome per line, space-separated genes
        f.write(' '.join(genes) + '\n')

print(f"   ✅ Saved: {output_file}")
print(f"   Format: One genome per line, space-separated genes")
print(f"   Total lines: {len(gene_orders):,}")
print()

# Save metadata
metadata = {
    'total_genomes': len(gene_orders),
    'min_genome_length': min_length,
    'max_genome_length': max_length,
    'avg_genome_length': float(avg_length),
    'median_genome_length': float(median_length),
    'recommended_max_seq_length': recommended_max,
    'input_file': output_file,
    'approach': 'james_original_exact',
    'note': 'Let panGPT.py handle splitting, padding, tokenization'
}

with open('data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("6. Metadata saved")
print()

print("="*100)
print("✅ INPUT FILE READY FOR JAMES'S CODE")
print("="*100)
print()

print("Next Step: Run James's panGPT.py")
print()
print("Recommended command:")
print(f"python panGPT.py \\")
print(f"  --input_file data/all_genomes.txt \\")
print(f"  --max_seq_length {recommended_max} \\")
print(f"  --train_size 0.8 \\")
print(f"  --val_size 0.1 \\")
print(f"  --batch_size 32 \\")
print(f"  --epochs 50 \\")
print(f"  --model_save_path checkpoints/james_model.pth \\")
print(f"  --tokenizer_file tokenizer.json")
print()
print("James's code will handle:")
print("  ✓ Train/val/test split (80/10/10)")
print("  ✓ Tokenization")
print("  ✓ Padding to max_seq_length")
print("  ✓ Training")
print()

