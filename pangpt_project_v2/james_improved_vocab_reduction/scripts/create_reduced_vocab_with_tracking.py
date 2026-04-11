#!/usr/bin/env python3
"""
Create reduced vocabulary dataset WITH detailed tracking of changes
"""
import argparse
from collections import Counter
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, 
                       default='../../james_exact_approach/data/all_genomes.txt',
                       help='Original genome file')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Minimum gene frequency to keep')
    parser.add_argument('--output_dir', type=str, default='../data',
                       help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*100)
    print("CREATING REDUCED VOCABULARY DATASET WITH CHANGE TRACKING")
    print("="*100)
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
    
    # Identify frequent and rare genes
    frequent_genes = {gene for gene, count in gene_counts.items() 
                     if count >= args.threshold}
    
    rare_genes = {gene for gene, count in gene_counts.items() 
                 if count < args.threshold}
    
    # Calculate coverage
    frequent_count = sum(count for gene, count in gene_counts.items() 
                        if count >= args.threshold)
    coverage = frequent_count / total_genes * 100
    
    print("\n" + "="*100)
    print("VOCABULARY REDUCTION SUMMARY")
    print("="*100)
    print(f"Original vocabulary: {len(gene_counts):,} genes")
    print(f"Frequent genes (≥{args.threshold}): {len(frequent_genes):,} genes")
    print(f"Rare genes (<{args.threshold}): {len(rare_genes):,} genes")
    print(f"Vocabulary reduction: {len(rare_genes):,} genes removed ({len(rare_genes)/len(gene_counts)*100:.1f}%)")
    print(f"Coverage: {coverage:.2f}% of all gene occurrences")
    print("="*100)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each genome and track changes
    print("\nProcessing genomes and tracking changes...")
    
    # Output files
    reduced_file = os.path.join(args.output_dir, f'all_genomes_vocab{args.threshold}.txt')
    tracking_file = os.path.join(args.output_dir, f'vocabulary_reduction_tracking_threshold{args.threshold}.txt')
    detailed_comparison_file = os.path.join(args.output_dir, f'detailed_comparison_threshold{args.threshold}.txt')
    
    # Statistics
    total_genes_removed = 0
    genomes_affected = 0
    removal_stats = []
    
    # Open all output files
    with open(reduced_file, 'w') as f_reduced, \
         open(tracking_file, 'w') as f_track, \
         open(detailed_comparison_file, 'w') as f_detail:
        
        # Write headers
        f_track.write("="*100 + "\n")
        f_track.write("VOCABULARY REDUCTION TRACKING REPORT\n")
        f_track.write("="*100 + "\n")
        f_track.write(f"Threshold: ≥{args.threshold} occurrences\n")
        f_track.write(f"Total genomes: {len(genomes)}\n")
        f_track.write(f"Genes kept: {len(frequent_genes):,}\n")
        f_track.write(f"Genes removed: {len(rare_genes):,}\n")
        f_track.write("="*100 + "\n\n")
        
        f_detail.write("="*100 + "\n")
        f_detail.write("DETAILED GENOME-BY-GENOME COMPARISON\n")
        f_detail.write("="*100 + "\n")
        f_detail.write(f"Format: Each genome shows ORIGINAL → REDUCED with removed genes listed\n")
        f_detail.write("="*100 + "\n\n")
        
        # Process each genome
        for genome_idx, genome in enumerate(genomes, 1):
            genes = genome.split()
            
            # Separate frequent and rare genes
            kept_genes = []
            removed_genes = []
            removed_positions = []
            
            for pos, gene in enumerate(genes, 1):
                if gene in frequent_genes:
                    kept_genes.append(gene)
                else:
                    kept_genes.append("<RARE>")
                    removed_genes.append((pos, gene, gene_counts[gene]))
                    removed_positions.append(pos)
            
            # Write reduced genome
            f_reduced.write(' '.join(kept_genes) + '\n')
            
            # Track if this genome was affected
            if removed_genes:
                genomes_affected += 1
                total_genes_removed += len(removed_genes)
                
                # Write to tracking file
                f_track.write(f"Genome {genome_idx}:\n")
                f_track.write(f"  Original length: {len(genes)} genes\n")
                f_track.write(f"  Genes removed: {len(removed_genes)} genes ({len(removed_genes)/len(genes)*100:.1f}%)\n")
                f_track.write(f"  Positions affected: {', '.join(map(str, removed_positions[:20]))}")
                if len(removed_positions) > 20:
                    f_track.write(f"... and {len(removed_positions)-20} more")
                f_track.write("\n")
                
                f_track.write(f"  Removed genes (showing first 10):\n")
                for pos, gene, count in removed_genes[:10]:
                    f_track.write(f"    Position {pos}: {gene} (appeared {count} times in dataset)\n")
                if len(removed_genes) > 10:
                    f_track.write(f"... and {len(removed_genes)-10} more\n")
                f_track.write("\n")
                
                # Write detailed comparison
                f_detail.write(f"{'='*100}\n")
                f_detail.write(f"GENOME {genome_idx}\n")
                f_detail.write(f"{'='*100}\n")
                f_detail.write(f"Original length: {len(genes)} genes | Removed: {len(removed_genes)} genes\n")
                f_detail.write(f"{'-'*100}\n")
                
                # Show original with markers
                f_detail.write("\nORIGINAL (with markers for removed genes):\n")
                marked_genes = []
                for pos, gene in enumerate(genes, 1):
                    if pos in removed_positions:
                        marked_genes.append(f"[{gene}]")  # Mark removed genes with brackets
                    else:
                        marked_genes.append(gene)
                
                # Write in chunks of 10 genes per line for readability
                for i in range(0, len(marked_genes), 10):
                    chunk = marked_genes[i:i+10]
                    f_detail.write(f"  {' '.join(chunk)}\n")
                
                f_detail.write(f"\nREDUCED (with <RARE> token):\n")
                for i in range(0, len(kept_genes), 10):
                    chunk = kept_genes[i:i+10]
                    f_detail.write(f"  {' '.join(chunk)}\n")
                
                f_detail.write(f"\nREMOVED GENES ({len(removed_genes)} total):\n")
                f_detail.write(f"{'Position':<10} {'Gene':<30} {'Frequency in Dataset':<25}\n")
                f_detail.write(f"{'-'*100}\n")
                for pos, gene, count in removed_genes:
                    f_detail.write(f"{pos:<10} {gene:<30} {count} occurrences\n")
                
                f_detail.write("\n\n")
                
                # Store stats
                removal_stats.append({
                    'genome_idx': genome_idx,
                    'total_genes': len(genes),
                    'removed_count': len(removed_genes),
                    'removed_percent': len(removed_genes)/len(genes)*100
                })
            
            # Progress indicator
            if genome_idx % 1000 == 0:
                print(f"  Processed {genome_idx}/{len(genomes)} genomes...")
        
        # Write summary statistics
        f_track.write("\n" + "="*100 + "\n")
        f_track.write("OVERALL STATISTICS\n")
        f_track.write("="*100 + "\n")
        f_track.write(f"Total genomes: {len(genomes)}\n")
        f_track.write(f"Genomes affected: {genomes_affected} ({genomes_affected/len(genomes)*100:.1f}%)\n")
        f_track.write(f"Total genes removed: {total_genes_removed:,}\n")
        f_track.write(f"Average genes removed per affected genome: {total_genes_removed/genomes_affected if genomes_affected > 0 else 0:.1f}\n")
        
        if removal_stats:
            avg_removal_percent = sum(s['removed_percent'] for s in removal_stats) / len(removal_stats)
            max_removal = max(removal_stats, key=lambda x: x['removed_percent'])
            min_removal = min(removal_stats, key=lambda x: x['removed_percent'])
            
            f_track.write(f"\nRemoval percentage statistics:\n")
            f_track.write(f"  Average: {avg_removal_percent:.2f}% of genes per genome\n")
            f_track.write(f"  Maximum: {max_removal['removed_percent']:.2f}% (Genome {max_removal['genome_idx']})\n")
            f_track.write(f"  Minimum: {min_removal['removed_percent']:.2f}% (Genome {min_removal['genome_idx']})\n")
        
        f_track.write("\n" + "="*100 + "\n")
        f_track.write("RARE GENES SUMMARY (Top 50 most common rare genes)\n")
        f_track.write("="*100 + "\n")
        f_track.write(f"{'Gene':<30} {'Occurrences':<15} {'Frequency':<15}\n")
        f_track.write("-"*100 + "\n")
        
        rare_sorted = [(g, c) for g, c in gene_counts.most_common() if g in rare_genes][:50]
        for gene, count in rare_sorted:
            freq = count / total_genes * 100
            f_track.write(f"{gene:<30} {count:<15} {freq:.6f}%\n")
        
        f_track.write("="*100 + "\n")
    
    print(f"✓ Processed all {len(genomes)} genomes")
    print(f"\n✓ Saved reduced vocabulary dataset to: {reduced_file}")
    print(f"✓ Saved tracking report to: {tracking_file}")
    print(f"✓ Saved detailed comparison to: {detailed_comparison_file}")
    
    # Create summary file
    summary_file = os.path.join(args.output_dir, f'reduction_summary_threshold{args.threshold}.txt')
    with open(summary_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("VOCABULARY REDUCTION SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Threshold: ≥{args.threshold} occurrences\n\n")
        f.write(f"VOCABULARY:\n")
        f.write(f"  Original: {len(gene_counts):,} unique genes\n")
        f.write(f"  Reduced:  {len(frequent_genes):,} unique genes\n")
        f.write(f"  Removed:  {len(rare_genes):,} genes ({len(rare_genes)/len(gene_counts)*100:.1f}%)\n\n")
        f.write(f"COVERAGE:\n")
        f.write(f"  Total gene occurrences: {total_genes:,}\n")
        f.write(f"  Covered by reduced vocab: {frequent_count:,} ({coverage:.2f}%)\n\n")
        f.write(f"GENOMES:\n")
        f.write(f"  Total genomes: {len(genomes)}\n")
        f.write(f"  Genomes with changes: {genomes_affected} ({genomes_affected/len(genomes)*100:.1f}%)\n")
        f.write(f"  Total gene replacements: {total_genes_removed:,}\n\n")
        f.write(f"FILES CREATED:\n")
        f.write(f"  1. {os.path.basename(reduced_file)}\n")
        f.write(f"     - Reduced vocabulary dataset (ready for training)\n")
        f.write(f"     - Rare genes replaced with <RARE> token\n\n")
        f.write(f"  2. {os.path.basename(tracking_file)}\n")
        f.write(f"     - Summary of changes per genome\n")
        f.write(f"     - Shows which genes were removed and from where\n\n")
        f.write(f"  3. {os.path.basename(detailed_comparison_file)}\n")
        f.write(f"     - Side-by-side comparison of original vs reduced\n")
        f.write(f"     - Shows exact positions of removed genes\n")
        f.write(f"     - Marked format: [gene] indicates removed gene\n\n")
        f.write("="*100 + "\n")
    
    print(f"✓ Saved summary to: {summary_file}")
    
    print("\n" + "="*100)
    print("✓ VOCABULARY REDUCTION COMPLETE WITH FULL TRACKING!")
    print("="*100)
    print(f"\nFiles created:")
    print(f"  1. Reduced dataset:      {reduced_file}")
    print(f"  2. Tracking report:      {tracking_file}")
    print(f"  3. Detailed comparison:  {detailed_comparison_file}")
    print(f"  4. Summary:              {summary_file}")
    print("="*100)

if __name__ == "__main__":
    main()
