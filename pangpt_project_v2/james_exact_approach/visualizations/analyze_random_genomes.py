#!/usr/bin/env python3
"""
Analyze 5 random test genomes using existing panPrompt.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import random
import argparse

# Import from existing panPrompt.py
from panPrompt import SimpleTransformerModel

def load_model(model_path, tokenizer_path, device='cpu'):
    """Load model and tokenizer"""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Initialize model with James' parameters
    model = SimpleTransformerModel(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_length=4096,
        dropout_rate=0.2,
        pe_max_len=5000,
        pe_dropout_rate=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def get_top_predictions(model, tokenizer, input_sequence, top_k=50, device='cpu'):
    """Get top-k predictions with probabilities"""
    model.eval()
    
    # Tokenize
    tokens = tokenizer.encode(input_sequence).ids
    
    # Truncate if too long
    if len(tokens) > 4095:
        tokens = tokens[-4095:]
    
    input_ids = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, top_k)
    
    # Convert to gene names
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        gene = id_to_token.get(idx, f"UNK_{idx}")
        predictions.append((gene, float(prob)))
    
    return predictions

def analyze_genome(genome_sequence, model, tokenizer, top_k=50, device='cpu'):
    """Analyze one genome"""
    genes = genome_sequence.split()
    
    if len(genes) < 2:
        return None
    
    # Use all but last gene as input
    input_genes = genes[:-1]
    actual_gene = genes[-1]
    input_seq = ' '.join(input_genes)
    
    # Get predictions
    predictions = get_top_predictions(model, tokenizer, input_seq, top_k, device)
    
    # Find actual gene rank
    actual_rank = None
    actual_prob = None
    for i, (gene, prob) in enumerate(predictions, 1):
        if gene == actual_gene:
            actual_rank = i
            actual_prob = prob
            break
    
    return {
        'length': len(genes),
        'actual_gene': actual_gene,
        'actual_rank': actual_rank,
        'actual_prob': actual_prob,
        'predictions': predictions
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tokenizer_path', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--num_genomes', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='visualizations/panprompt_results.txt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("="*80)
    print("PAN-PROMPTING: RANDOM TEST GENOME ANALYSIS")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.device)
    print(f"✓ Model loaded")
    print(f"✓ Vocab size: {tokenizer.get_vocab_size()}")
    
    # Load test genomes
    print(f"\nLoading test genomes from: {args.test_file}")
    with open(args.test_file) as f:
        genomes = [line.strip() for line in f if line.strip()]
    print(f"✓ Loaded {len(genomes)} genomes")
    
    # Select random genomes
    selected = random.sample(genomes, min(args.num_genomes, len(genomes)))
    print(f"✓ Selected {len(selected)} random genomes\n")
    
    # Analyze each
    results = []
    output_lines = []
    
    output_lines.append("="*80)
    output_lines.append("PAN-PROMPTING RESULTS")
    output_lines.append("="*80 + "\n")
    
    for i, genome in enumerate(selected, 1):
        print(f"\n{'='*80}")
        print(f"GENOME {i}/{len(selected)}")
        print(f"{'='*80}")
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"GENOME {i}/{len(selected)}")
        output_lines.append(f"{'='*80}")
        
        result = analyze_genome(genome, model, tokenizer, args.top_k, args.device)
        
        if not result:
            print("  Skipped (too short)")
            continue
        
        results.append(result)
        
        print(f"  Length: {result['length']} genes")
        print(f"  Actual: {result['actual_gene']}")
        
        output_lines.append(f"Length: {result['length']} genes")
        output_lines.append(f"Actual next gene: {result['actual_gene']}")
        
        if result['actual_rank']:
            msg = f"✓ Found at rank {result['actual_rank']} (prob: {result['actual_prob']*100:.2f}%)"
            print(f"  {msg}")
            output_lines.append(msg)
        else:
            msg = f"✗ Not in top {args.top_k}"
            print(f"  {msg}")
            output_lines.append(msg)
        
        # Show top 20
        print(f"\n  {'Rank':<6} {'Gene':<30} {'Prob%':<10} {'Cumul%':<10}")
        print(f"  {'-'*56}")
        
        output_lines.append(f"\n{'Rank':<6} {'Gene':<30} {'Prob%':<10} {'Cumul%':<10}")
        output_lines.append("-"*56)
        
        cumul = 0
        for rank, (gene, prob) in enumerate(result['predictions'][:20], 1):
            cumul += prob
            marker = " ✓" if gene == result['actual_gene'] else ""
            line = f"  {rank:<6} {gene:<30} {prob*100:>6.2f}    {cumul*100:>6.2f}{marker}"
            print(line)
            output_lines.append(line.strip())
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    output_lines.append(f"\n{'='*80}")
    output_lines.append("SUMMARY")
    output_lines.append(f"{'='*80}")
    
    total = len(results)
    top1 = sum(1 for r in results if r['actual_rank'] == 1)
    top5 = sum(1 for r in results if r['actual_rank'] and r['actual_rank'] <= 5)
    top10 = sum(1 for r in results if r['actual_rank'] and r['actual_rank'] <= 10)
    top50 = sum(1 for r in results if r['actual_rank'] and r['actual_rank'] <= 50)
    
    ranks = [r['actual_rank'] for r in results if r['actual_rank']]
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    
    summary = f"""
Total genomes: {total}
Top-1 accuracy: {top1}/{total} ({top1/total*100:.1f}%)
Top-5 accuracy: {top5}/{total} ({top5/total*100:.1f}%)
Top-10 accuracy: {top10}/{total} ({top10/total*100:.1f}%)
Top-50 accuracy: {top50}/{total} ({top50/total*100:.1f}%)
Avg rank (when found): {avg_rank:.1f}
"""
    
    print(summary)
    output_lines.append(summary)
    output_lines.append("="*80)
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Saved to: {args.output}")
    print("="*80)

if __name__ == "__main__":
    main()
