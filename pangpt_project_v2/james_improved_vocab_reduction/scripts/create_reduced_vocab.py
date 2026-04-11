#!/usr/bin/env python3
"""
Create reduced vocabulary dataset
"""
import argparse
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, 
                       default='../james_exact_approach/data/all_genomes.txt',
                       help='Original genome file')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Minimum gene frequency to keep')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print("CREATING REDUCED VOCABULARY DATASET")
    print("="*80)
    print(f"Input file: {args.input_file}")
    print(f"Frequency threshold: ≥{args.threshold} occurrences")
    print(f"Output directory: {args.output_dir}")
    
    # Load genomes
    print("\nLoading genomes...")
    with open(args.input_file, 'r') as f:
        genomes = [line.strip() for line in f.readlines()]
    
    print(f"✓ Loaded {len(genomes)} genomes")
    
    # Count gene frequencies
    print("\nCounting gene frequencies...")
    all_genes = []
    for genome in genomes:
        all_genes.extend(genome.split())
    
    gene_counts = Counter(all_genes)
    total_genes = len(all_genes)
    
    print(f"✓ Total gene occurrences: {total_genes:,}")
    print(f"✓ Unique genes: {len(gene_counts):,}")
    
    # Identify frequent genes
    frequent_genes = {gene for gene, count in gene_counts.items() 
                     if count >= args.threshold}
    
    rare_genes = {gene for gene, count in gene_counts.items() 
                 if count < args.threshold}
    
    # Calculate coverage
    frequent_count = sum(count for gene, count in gene_counts.items() 
                        if count >= args.threshold)
    coverage = frequent_count / total_genes * 100
    
    print("\n" + "="*80)
    print("VOCABULARY REDUCTION SUMMARY")
    print("="*80)
    print(f"Original vocabulary: {len(gene_counts):,} genes")
    print(f"Frequent genes (≥{args.threshold}): {len(frequent_genes):,} genes")
    print(f"Rare genes (<{args.threshold}): {len(rare_genes):,} genes")
    print(f"Vocabulary reduction: {len(rare_genes):,} genes removed ({len(rare_genes)/len(gene_counts)*100:.1f}%)")
    print(f"Coverage: {coverage:.2f}% of all gene occurrences")
    print("="*80)
    
    # Map rare genes to <RARE>
    print("\nMapping rare genes to <RARE> token...")
    
    def map_rare_genes(genome_text):
        genes = genome_text.split()
        mapped_genes = []
        for gene in genes:
            if gene in frequent_genes:
                mapped_genes.append(gene)
            else:
                mapped_genes.append("<RARE>")
        return ' '.join(mapped_genes)
    
    # Create output filename
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'all_genomes_vocab{args.threshold}.txt')
    
    # Save reduced dataset
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        for genome in genomes:
            reduced = map_rare_genes(genome)
            f.write(reduced + '\n')
    
    print(f"✓ Saved reduced vocabulary dataset")
    
    # Save vocabulary info
    vocab_info_file = os.path.join(args.output_dir, f'vocab_info_threshold{args.threshold}.txt')
    with open(vocab_info_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VOCABULARY REDUCTION INFO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Threshold: ≥{args.threshold} occurrences\n")
        f.write(f"Original vocabulary: {len(gene_counts):,} genes\n")
        f.write(f"Reduced vocabulary: {len(frequent_genes):,} genes\n")
        f.write(f"Genes mapped to <RARE>: {len(rare_genes):,}\n")
        f.write(f"Coverage: {coverage:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOP 50 MOST FREQUENT GENES (KEPT)\n")
        f.write("="*80 + "\n")
        for i, (gene, count) in enumerate(gene_counts.most_common(50), 1):
            if gene in frequent_genes:
                f.write(f"{i:3d}. {gene:30s} {count:10,d} ({count/total_genes*100:6.4f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SAMPLE RARE GENES (MAPPED TO <RARE>)\n")
        f.write("="*80 + "\n")
        rare_samples = [(g, c) for g, c in gene_counts.items() if g in rare_genes][:50]
        for gene, count in rare_samples:
            f.write(f"{gene:30s} {count:10,d} ({count/total_genes*100:6.6f}%)\n")
    
    print(f"✓ Saved vocabulary info to: {vocab_info_file}")
    
    # Create statistics summary
    stats = {
        'threshold': args.threshold,
        'original_vocab': len(gene_counts),
        'reduced_vocab': len(frequent_genes),
        'rare_genes': len(rare_genes),
        'coverage': coverage,
        'total_occurrences': total_genes,
        'frequent_occurrences': frequent_count
    }
    
    import json
    stats_file = os.path.join(args.output_dir, f'vocab_stats_threshold{args.threshold}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved statistics to: {stats_file}")
    
    print("\n" + "="*80)
    print("✓ VOCABULARY REDUCTION COMPLETE!")
    print("="*80)
    print(f"\nNext step: Train model with reduced vocabulary dataset:")
    print(f"  python train_reduced_vocab.py --input_file {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
