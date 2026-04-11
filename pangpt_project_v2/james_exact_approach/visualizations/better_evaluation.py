#!/usr/bin/env python3
"""
Better evaluation: Test on multiple positions, not just last gene
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import random
from panPrompt import SimpleTransformerModel

def load_model():
    tokenizer = Tokenizer.from_file('tokenizer.json')
    model = SimpleTransformerModel(
        vocab_size=70000, embed_dim=512, num_heads=8, num_layers=6,
        max_seq_length=4096, dropout_rate=0.2, pe_max_len=5000, pe_dropout_rate=0.1
    )
    checkpoint = torch.load('checkpoints/james_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer

def predict_at_position(model, tokenizer, genes, position):
    """Predict gene at specific position"""
    input_genes = genes[:position]
    actual_gene = genes[position]
    
    # Truncate if too long
    if len(input_genes) > 4095:
        input_genes = input_genes[-4095:]
    
    input_seq = ' '.join(input_genes)
    tokens = tokenizer.encode(input_seq).ids
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 100)
    
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Find actual gene rank
    actual_rank = None
    for i, idx in enumerate(top_indices.cpu().numpy(), 1):
        gene = id_to_token.get(idx, f"UNK_{idx}")
        if gene == actual_gene:
            actual_rank = i
            break
    
    return actual_rank

print("="*80)
print("BETTER EVALUATION: Multiple positions per genome")
print("="*80)

model, tokenizer = load_model()

# Load test genomes
with open('data/all_genomes.txt') as f:
    genomes = [line.strip() for line in f]

random.seed(42)
selected = random.sample(genomes, 10)  # Test 10 genomes

results = {
    'top1': 0,
    'top5': 0,
    'top10': 0,
    'top50': 0,
    'top100': 0,
    'total': 0
}

print("\nTesting 10 random genomes, 10 positions each...\n")

for genome_idx, genome in enumerate(selected, 1):
    genes = genome.split()
    
    # Test 10 random positions in this genome
    test_positions = random.sample(range(100, len(genes)), min(10, len(genes)-100))
    
    print(f"Genome {genome_idx} ({len(genes)} genes):")
    
    for pos in test_positions:
        rank = predict_at_position(model, tokenizer, genes, pos)
        results['total'] += 1
        
        if rank:
            if rank == 1:
                results['top1'] += 1
            if rank <= 5:
                results['top5'] += 1
            if rank <= 10:
                results['top10'] += 1
            if rank <= 50:
                results['top50'] += 1
            if rank <= 100:
                results['top100'] += 1
            
            status = "✓" if rank <= 10 else "○"
            print(f"  Position {pos}: {status} Rank {rank}")
        else:
            print(f"  Position {pos}: ✗ Not in top 100")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total predictions: {results['total']}")
print(f"Top-1 accuracy:   {results['top1']}/{results['total']} ({results['top1']/results['total']*100:.1f}%)")
print(f"Top-5 accuracy:   {results['top5']}/{results['total']} ({results['top5']/results['total']*100:.1f}%)")
print(f"Top-10 accuracy:  {results['top10']}/{results['total']} ({results['top10']/results['total']*100:.1f}%)")
print(f"Top-50 accuracy:  {results['top50']}/{results['total']} ({results['top50']/results['total']*100:.1f}%)")
print(f"Top-100 accuracy: {results['top100']}/{results['total']} ({results['top100']/results['total']*100:.1f}%)")
print("="*80)
