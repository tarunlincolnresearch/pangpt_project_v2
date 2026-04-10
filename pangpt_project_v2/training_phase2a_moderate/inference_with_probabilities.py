#!/usr/bin/env python3
"""
Inference script to predict next genes with probability distributions
Based on James's panPrompt but with detailed probability analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
import math

# ============================================================
# Model Architecture (same as training)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate, pe_max_len, pe_dropout_rate):
        super(SimpleTransformerModel, self).__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=pe_dropout_rate)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.out(x)

# ============================================================
# Inference Functions
# ============================================================

def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    """Load trained model and tokenizer"""
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Initialize model (same architecture as training)
    model = SimpleTransformerModel(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_length=512,
        dropout_rate=0.15,
        pe_max_len=5000,
        pe_dropout_rate=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Vocab size: {vocab_size}")
    
    return model, tokenizer

def predict_next_gene_with_probabilities(model, tokenizer, prompt, device, top_k=20):
    """
    Predict next gene with full probability distribution
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: String of space-separated genes
        device: CPU or CUDA
        top_k: Number of top predictions to return
    
    Returns:
        List of (gene, probability) tuples
    """
    
    # Encode prompt
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids)
        
        # Get logits for the last position (next token prediction)
        last_token_logits = outputs[0, -1, :]
        
        # Convert to probabilities
        probabilities = F.softmax(last_token_logits, dim=0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to gene names
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            gene = tokenizer.id_to_token(int(idx))
            results.append({
                'gene': gene,
                'probability': float(prob),
                'percentage': float(prob) * 100
            })
    
    return results

def analyze_prompt(model, tokenizer, prompt, device, top_k=20):
    """Comprehensive analysis of a prompt"""
    
    print(f"\n{'='*80}")
    print(f"PROMPT ANALYSIS")
    print(f"{'='*80}")
    print(f"\nInput prompt: {prompt}")
    print(f"Number of genes in prompt: {len(prompt.split())}")
    
    # Get predictions
    predictions = predict_next_gene_with_probabilities(model, tokenizer, prompt, device, top_k)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"TOP {top_k} NEXT GENE PREDICTIONS")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Gene':<20} {'Probability':<12} {'Percentage':<12}")
    print(f"{'-'*80}")
    
    for i, pred in enumerate(predictions, 1):
        print(f"{i:<6} {pred['gene']:<20} {pred['probability']:<12.6f} {pred['percentage']:<12.2f}%")
    
    # Probability distribution analysis
    print(f"\n{'='*80}")
    print(f"PROBABILITY DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    top_1_prob = predictions[0]['percentage']
    top_5_prob = sum(p['percentage'] for p in predictions[:5])
    top_10_prob = sum(p['percentage'] for p in predictions[:10])
    top_20_prob = sum(p['percentage'] for p in predictions[:20])
    
    print(f"Top 1 gene probability:  {top_1_prob:>6.2f}%")
    print(f"Top 5 genes probability: {top_5_prob:>6.2f}%")
    print(f"Top 10 genes probability: {top_10_prob:>6.2f}%")
    print(f"Top 20 genes probability: {top_20_prob:>6.2f}%")
    
    # Entropy analysis
    import numpy as np
    probs = np.array([p['probability'] for p in predictions])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(predictions))
    normalized_entropy = entropy / max_entropy
    
    print(f"\nEntropy: {entropy:.2f} bits")
    print(f"Normalized entropy: {normalized_entropy:.2f} (0=certain, 1=uniform)")
    
    if normalized_entropy > 0.9:
        print("⚠️  Model is very uncertain (nearly uniform distribution)")
    elif normalized_entropy < 0.3:
        print("✅ Model is confident in its predictions")
    else:
        print("ℹ️  Model has moderate confidence")
    
    return predictions

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with probability analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, required=True, help="Input gene sequence")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top predictions to show")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)
    
    # Analyze prompt
    predictions = analyze_prompt(model, tokenizer, args.prompt, device, args.top_k)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

