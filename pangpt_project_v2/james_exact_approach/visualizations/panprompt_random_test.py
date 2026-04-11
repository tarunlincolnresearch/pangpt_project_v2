#!/usr/bin/env python3
"""
Pan-prompting on 5 random test genomes
"""
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load trained model and tokenizer"""
    from train_deep import SimpleTransformerModel
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def predict_next_gene(model, tokenizer, genome_sequence, top_k=50, device='cpu'):
    """Predict next gene with probabilities"""
    model.to(device)
    model.eval()
    
    encoded = tokenizer.encode(genome_sequence).ids
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_gene_logits = outputs[0, -1, :]
        probabilities = F.softmax(next_gene_logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        gene_name = id_to_token.get(idx, f"<UNK_{idx}>")
        predictions.append((gene_name, float(prob)))
    
    return predictions

def analyze_genome(genome_id, genome_sequence, model, tokenizer, top_k=50, device='cpu'):
    """Analyze a single genome"""
    genes = genome_sequence.split()
    
    if len(genes) < 2:
        print(f"  ✗ Genome too short ({len(genes)} genes)")
        return None
    
    input_genes = genes[:-1]
    actual_next_gene = genes[-1]
    input_sequence = ' '.join(input_genes)
    
    predictions = predict_next_gene(model, tokenizer, input_sequence, top_k=top_k, device=device)
    
    # Find actual gene rank
    actual_rank = None
    actual_prob = None
    for i, (gene, prob) in enumerate(predictions, 1):
        if gene == actual_next_gene:
            actual_rank = i
            actual_prob = prob
            break
    
    return {
        'genome_id': genome_id,
        'genome_length': len(genes),
        'actual_gene': actual_next_gene,
        'actual_rank': actual_rank,
        'actual_prob': actual_prob,
        'predictions': predictions
    }

def main():
    parser = argparse.ArgumentParser(description='Pan-prompting on random test genomes')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--test_file', type=str, required=True, help='Test genomes file')
    parser.add_argument('--num_genomes', type=int, default=5, help='Number of random genomes to test')
    parser.add_argument('--top_k', type=int, default=50, help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--output_file', type=str, default='panprompt_results.txt', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("="*80)
    print("PAN-PROMPTING: RANDOM TEST GENOME ANALYSIS")
    print("="*80)
    
    # Load model
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    print(f"✓ Model loaded")
    print(f"✓ Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Load test genomes
    print(f"\nLoading test genomes from: {args.test_file}")
    with open(args.test_file, 'r') as f:
        test_genomes = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"✓ Loaded {len(test_genomes)} test genomes")
    
    # Select random genomes
    if len(test_genomes) < args.num_genomes:
        print(f"Warning: Only {len(test_genomes)} genomes available, using all")
        selected_genomes = test_genomes
    else:
        selected_genomes = random.sample(test_genomes, args.num_genomes)
    
    print(f"✓ Selected {len(selected_genomes)} random genomes")
    print()
    
    # Analyze each genome
    results = []
    output_lines = []
    
    output_lines.append("="*80)
    output_lines.append("PAN-PROMPTING RESULTS: RANDOM TEST GENOMES")
    output_lines.append("="*80)
    output_lines.append("")
    
    for i, genome in enumerate(selected_genomes, 1):
        print(f"\n{'='*80}")
        print(f"GENOME {i}/{len(selected_genomes)}")
        print(f"{'='*80}")
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"GENOME {i}/{len(selected_genomes)}")
        output_lines.append(f"{'='*80}")
        
        result = analyze_genome(i, genome, model, tokenizer, top_k=args.top_k, device=args.device)
        
        if result is None:
            continue
        
        results.append(result)
        
        print(f"Genome length: {result['genome_length']} genes")
        print(f"Actual next gene: {result['actual_gene']}")
        
        output_lines.append(f"Genome length: {result['genome_length']} genes")
        output_lines.append(f"Actual next gene: {result['actual_gene']}")
        
        if result['actual_rank']:
            print(f"✓ Actual gene found at rank {result['actual_rank']}")
            print(f"  Probability: {result['actual_prob']*100:.2f}%")
            output_lines.append(f"✓ Actual gene found at rank {result['actual_rank']}")
            output_lines.append(f"  Probability: {result['actual_prob']*100:.2f}%")
        else:
            print(f"✗ Actual gene not in top {args.top_k} predictions")
            output_lines.append(f"✗ Actual gene not in top {args.top_k} predictions")
        
        print(f"\nTop {min(20, args.top_k)} predictions:")
        print(f"{'Rank':<6} {'Gene':<30} {'Probability':<15} {'Cumulative':<15}")
        print("-"*80)
        
        output_lines.append(f"\nTop {min(20, args.top_k)} predictions:")
        output_lines.append(f"{'Rank':<6} {'Gene':<30} {'Probability':<15} {'Cumulative':<15}")
        output_lines.append("-"*80)
        
        cumulative = 0.0
        for rank, (gene, prob) in enumerate(result['predictions'][:20], 1):
            cumulative += prob
            marker = " ✓ ACTUAL" if gene == result['actual_gene'] else ""
            
            line = f"{rank:<6} {gene:<30} {prob*100:>6.2f}%        {cumulative*100:>6.2f}%{marker}"
            print(line)
            output_lines.append(line)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    output_lines.append(f"\n{'='*80}")
    output_lines.append("SUMMARY STATISTICS")
    output_lines.append(f"{'='*80}")
    
    total = len(results)
    found_in_top1 = sum(1 for r in results if r['actual_rank'] == 1)
    found_in_top5 = sum(1 for r in results if r['actual_rank'] and r['actual_rank'] <= 5)
    found_in_top10 = sum(1 for r in results if r['actual_rank'] and r['actual_rank'] <= 10)
    found_in_top50 = sum(1 for r in results if r['actual_rank'] and r['actual_rank'] <= 50)
    
    avg_rank = sum(r['actual_rank'] for r in results if r['actual_rank']) / sum(1 for r in results if r['actual_rank']) if any(r['actual_rank'] for r in results) else 0
    
    summary = f"""
Total genomes analyzed: {total}
Top-1 accuracy: {found_in_top1}/{total} ({found_in_top1/total*100:.1f}%)
Top-5 accuracy: {found_in_top5}/{total} ({found_in_top5/total*100:.1f}%)
Top-10 accuracy: {found_in_top10}/{total} ({found_in_top10/total*100:.1f}%)
Top-50 accuracy: {found_in_top50}/{total} ({found_in_top50/total*100:.1f}%)
Average rank (when found): {avg_rank:.1f}
"""
    
    print(summary)
    output_lines.append(summary)
    output_lines.append("="*80)
    
    # Save results
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Results saved to: {args.output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
