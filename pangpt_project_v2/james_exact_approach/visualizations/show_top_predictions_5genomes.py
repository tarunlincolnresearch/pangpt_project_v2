#!/usr/bin/env python3
"""
Show top predictions with probabilities for 5 random genomes
Save detailed output
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

def get_predictions_with_probs(model, tokenizer, input_sequence, top_k=100):
    """Get top-k predictions with probabilities"""
    tokens = tokenizer.encode(input_sequence).ids
    
    # Truncate if too long
    if len(tokens) > 4095:
        tokens = tokens[-4095:]
    
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
    
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        gene = id_to_token.get(idx, f"UNK_{idx}")
        predictions.append((gene, float(prob)))
    
    return predictions

def format_genome_sequence(genes, max_display=50):
    """Format genome sequence for display"""
    if len(genes) <= max_display:
        return ' '.join(genes)
    else:
        first_25 = ' '.join(genes[:25])
        last_25 = ' '.join(genes[-25:])
        return f"{first_25}... [{len(genes)-50} genes omitted]... {last_25}"

print("="*100)
print("TOP PREDICTIONS WITH PROBABILITIES FOR 5 RANDOM GENOMES")
print("="*100)

model, tokenizer = load_model()
print("✓ Model loaded")

# Load test genomes
with open('data/all_genomes.txt') as f:
    genomes = [line.strip() for line in f]

random.seed(42)
selected = random.sample(genomes, 5)

output_lines = []
output_lines.append("="*100)
output_lines.append("DETAILED PREDICTIONS FOR 5 RANDOM TEST GENOMES")
output_lines.append("="*100)
output_lines.append("")

for genome_idx, genome in enumerate(selected, 1):
    genes = genome.split()
    
    if len(genes) < 2:
        continue
    
    input_genes = genes[:-1]
    actual_gene = genes[-1]
    input_seq = ' '.join(input_genes)
    
    print(f"\n{'='*100}")
    print(f"GENOME {genome_idx}/5")
    print(f"{'='*100}")
    
    output_lines.append(f"\n{'='*100}")
    output_lines.append(f"GENOME {genome_idx}/5")
    output_lines.append(f"{'='*100}")
    
    # Show genome info
    print(f"Total genes in genome: {len(genes)}")
    print(f"Input sequence length: {len(input_genes)} genes")
    print(f"Actual next gene: {actual_gene}")
    print(f"\nInput sequence (showing first 25 and last 25 genes):")
    print(f"  {format_genome_sequence(input_genes)}")
    
    output_lines.append(f"Total genes in genome: {len(genes)}")
    output_lines.append(f"Input sequence length: {len(input_genes)} genes")
    output_lines.append(f"Actual next gene: {actual_gene}")
    output_lines.append(f"\nInput sequence (first 25 and last 25 genes):")
    output_lines.append(f"  {format_genome_sequence(input_genes)}")
    
    # Get predictions
    print(f"\nGenerating predictions...")
    predictions = get_predictions_with_probs(model, tokenizer, input_seq, top_k=100)
    
    # Find actual gene rank
    actual_rank = None
    actual_prob = None
    for i, (gene, prob) in enumerate(predictions, 1):
        if gene == actual_gene:
            actual_rank = i
            actual_prob = prob
            break
    
    print(f"\n{'─'*100}")
    print(f"PREDICTION RESULTS")
    print(f"{'─'*100}")
    
    output_lines.append(f"\n{'─'*100}")
    output_lines.append(f"PREDICTION RESULTS")
    output_lines.append(f"{'─'*100}")
    
    if actual_rank:
        status = f"✓ FOUND at rank {actual_rank} with probability {actual_prob*100:.4f}%"
        print(status)
        output_lines.append(status)
    else:
        status = f"✗ NOT FOUND in top 100 predictions"
        print(status)
        output_lines.append(status)
    
    # Show top 50 predictions
    print(f"\nTop 50 Predictions (out of 70,000 possible genes):")
    print(f"{'─'*100}")
    print(f"{'Rank':<6} {'Gene Name':<35} {'Probability':<15} {'Cumulative':<15} {'Status':<10}")
    print(f"{'─'*100}")
    
    output_lines.append(f"\nTop 50 Predictions (out of 70,000 possible genes):")
    output_lines.append(f"{'─'*100}")
    output_lines.append(f"{'Rank':<6} {'Gene Name':<35} {'Probability':<15} {'Cumulative':<15} {'Status':<10}")
    output_lines.append(f"{'─'*100}")
    
    cumulative = 0.0
    for rank, (gene, prob) in enumerate(predictions[:50], 1):
        cumulative += prob
        
        status = ""
        if gene == actual_gene:
            status = "← ACTUAL"
        
        line = f"{rank:<6} {gene:<35} {prob*100:>6.4f}%        {cumulative*100:>6.2f}%        {status}"
        print(line)
        output_lines.append(line)
    
    print(f"{'─'*100}")
    output_lines.append(f"{'─'*100}")
    
    # Show probability distribution summary
    top10_prob = sum(p for _, p in predictions[:10])
    top50_prob = sum(p for _, p in predictions[:50])
    top100_prob = sum(p for _, p in predictions[:100])
    
    summary = f"""
Probability Distribution Summary:
  - Top 10 genes account for:  {top10_prob*100:.2f}% of total probability
  - Top 50 genes account for:  {top50_prob*100:.2f}% of total probability
  - Top 100 genes account for: {top100_prob*100:.2f}% of total probability
  - Remaining 69,900 genes:    {(1-top100_prob)*100:.2f}% of total probability
"""
    print(summary)
    output_lines.append(summary)

# Save to file
output_file = 'panprompt_outputs/detailed_predictions_5genomes.txt'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"\n{'='*100}")
print(f"✓ Detailed output saved to: {output_file}")
print(f"{'='*100}")
