#!/usr/bin/env python3
"""
Pan-prompting: Show next gene predictions with probabilities
"""
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load trained model and tokenizer"""
    from train_deep import SimpleTransformerModel
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model (adjust these parameters based on your training)
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

def predict_next_gene(model, tokenizer, genome_sequence, top_k=10, device='cpu'):
    """
    Predict next gene with probabilities
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        genome_sequence: String of space-separated genes
        top_k: Number of top predictions to return
        device: Device to run on
    
    Returns:
        List of (gene_name, probability) tuples
    """
    model.to(device)
    model.eval()
    
    # Tokenize input
    encoded = tokenizer.encode(genome_sequence).ids
    
    # Create input tensor
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids)
        
        # Get logits for the last position (next gene prediction)
        next_gene_logits = outputs[0, -1, :]
        
        # Convert to probabilities
        probabilities = F.softmax(next_gene_logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Convert to gene names
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        gene_name = id_to_token.get(idx, f"<UNK_{idx}>")
        predictions.append((gene_name, float(prob)))
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Pan-prompting: Predict next gene')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--genome_file', type=str, required=True, help='File containing genome sequence')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    print("="*80)
    print("PAN-PROMPTING: NEXT GENE PREDICTION")
    print("="*80)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    print(f"✓ Model loaded from: {args.model_path}")
    print(f"✓ Tokenizer loaded from: {args.tokenizer_path}")
    print(f"✓ Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Load genome sequence
    print(f"\nLoading genome from: {args.genome_file}")
    with open(args.genome_file, 'r') as f:
        full_genome = f.readline().strip()
    
    genes = full_genome.split()
    print(f"✓ Genome length: {len(genes)} genes")
    
    # Use all but last gene as input
    input_genes = genes[:-1]
    actual_next_gene = genes[-1]
    
    input_sequence = ' '.join(input_genes)
    
    print(f"\nInput: {len(input_genes)} genes")
    print(f"Actual next gene: {actual_next_gene}")
    print(f"\nPredicting...")
    
    # Get predictions
    predictions = predict_next_gene(model, tokenizer, input_sequence, top_k=args.top_k, device=args.device)
    
    # Display results
    print("\n" + "="*80)
    print("TOP PREDICTIONS (with probabilities)")
    print("="*80)
    print(f"{'Rank':<6} {'Gene':<30} {'Probability':<15} {'Cumulative':<15}")
    print("-"*80)
    
    cumulative = 0.0
    actual_found = False
    actual_rank = None
    
    for i, (gene, prob) in enumerate(predictions, 1):
        cumulative += prob
        marker = " ✓ ACTUAL" if gene == actual_next_gene else ""
        
        if gene == actual_next_gene:
            actual_found = True
            actual_rank = i
        
        print(f"{i:<6} {gene:<30} {prob*100:>6.2f}%        {cumulative*100:>6.2f}%{marker}")
    
    print("="*80)
    
    if actual_found:
        print(f"\n✓ Actual gene '{actual_next_gene}' found at rank {actual_rank}")
        print(f"  Probability: {predictions[actual_rank-1][1]*100:.2f}%")
    else:
        print(f"\n✗ Actual gene '{actual_next_gene}' not in top {args.top_k} predictions")
        print(f"  Searching in full vocabulary...")
        
        # Find actual gene in full predictions
        all_predictions = predict_next_gene(model, tokenizer, input_sequence, 
                                           top_k=tokenizer.get_vocab_size(), device=args.device)
        for i, (gene, prob) in enumerate(all_predictions, 1):
            if gene == actual_next_gene:
                print(f"  Found at rank {i} with probability {prob*100:.4f}%")
                break
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
