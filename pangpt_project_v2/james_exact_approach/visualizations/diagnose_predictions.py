#!/usr/bin/env python3
"""
Diagnose why predictions are all the same
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
import torch
from panPrompt import SimpleTransformerModel

# Load tokenizer
tokenizer = Tokenizer.from_file('tokenizer.json')
vocab = tokenizer.get_vocab()

# Load model
model = SimpleTransformerModel(
    vocab_size=70000,
    embed_dim=512,
    num_heads=8,
    num_layers=6,
    max_seq_length=4096,
    dropout_rate=0.2,
    pe_max_len=5000,
    pe_dropout_rate=0.1
)

checkpoint = torch.load('checkpoints/james_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test genomes
with open('data/all_genomes.txt') as f:
    genomes = [line.strip() for line in f]

print("="*80)
print("DIAGNOSTIC: Checking actual genes in test set")
print("="*80)

# Check 5 random genomes
import random
random.seed(42)
selected = random.sample(genomes, 5)

actual_genes = []
for i, genome in enumerate(selected, 1):
    genes = genome.split()
    actual_gene = genes[-1]
    actual_genes.append(actual_gene)
    
    # Check if gene is in vocabulary
    gene_id = vocab.get(actual_gene, None)
    
    print(f"\nGenome {i}:")
    print(f"  Actual gene: {actual_gene}")
    print(f"  In vocabulary: {'YES' if gene_id is not None else 'NO'}")
    if gene_id is not None:
        print(f"  Gene ID: {gene_id}")

# Count frequency of these genes in training data
print("\n" + "="*80)
print("Checking gene frequencies in dataset")
print("="*80)

all_genes = []
for genome in genomes:
    all_genes.extend(genome.split())

from collections import Counter
gene_counts = Counter(all_genes)

for gene in actual_genes:
    count = gene_counts[gene]
    total = len(all_genes)
    freq = count / total * 100
    print(f"{gene}: appears {count} times ({freq:.4f}% of all genes)")

# Compare with top predicted genes
print("\n" + "="*80)
print("Top predicted genes frequency")
print("="*80)

top_predicted = ['stpA', 'papC_1', 'ung', 'group_30197', 'nusB']
for gene in top_predicted:
    count = gene_counts[gene]
    freq = count / total * 100
    print(f"{gene}: appears {count} times ({freq:.4f}% of all genes)")

print("\n" + "="*80)
