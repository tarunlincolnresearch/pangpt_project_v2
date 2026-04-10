import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
from torch import nn
import math
import pandas as pd

def print_banner():
    banner = """
    **************************************************
    *                                                *
    *    panPrompt Analysis - Probability Explorer   *
    *    v0.04 - Gene Prediction Analysis            *
    *                                                *
    **************************************************
    """
    print(banner)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :, :].transpose(0, 1)
        return self.dropout(x)

class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate=0.1, pe_max_len=5000, pe_dropout_rate=0.1):
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

def get_next_token_probabilities(model, tokenizer, prompt, top_k=20):
    """Get the top-k most likely next tokens with their probabilities"""
    model.eval()
    device = next(model.parameters()).device
    
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0, -1, :]
        probabilities = F.softmax(logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        gene_name = tokenizer.decode([int(idx)])
        results.append({
            'gene': gene_name,
            'probability': float(prob),
            'percentage': float(prob) * 100
        })
    
    return results

def find_preceding_genes(model, tokenizer, target_gene, context_genes, top_k=20):
    """Given a target gene, find which genes are most likely to precede it"""
    model.eval()
    device = next(model.parameters()).device
    
    target_tokens = tokenizer.encode(target_gene).ids
    if not target_tokens:
        return []
    
    target_token_id = target_tokens[0]
    
    results = []
    
    for gene in context_genes:
        tokens = tokenizer.encode(gene).ids
        if not tokens:
            continue
            
        input_ids = torch.tensor([tokens]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[0, -1, :]
            probabilities = F.softmax(logits, dim=-1)
        
        prob = probabilities[target_token_id].item()
        
        results.append({
            'preceding_gene': gene,
            'probability': prob,
            'percentage': prob * 100
        })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results[:top_k]

def read_prompt_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read().strip()
    return prompt

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Analyze gene prediction probabilities")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['next', 'preceding', 'both'], default='both')
    parser.add_argument("--target_gene", type=str, help="Target gene for 'preceding' mode")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    model = SimpleTransformerModel(
        vocab_size, args.embed_dim, args.num_heads, args.num_layers,
        args.max_seq_length, dropout_rate=0.1, pe_max_len=5000, pe_dropout_rate=0.1
    )
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    
    prompt = read_prompt_file(args.prompt_file)
    print(f"\nPrompt: {prompt}\n")
    
    if args.mode in ['next', 'both']:
        print("=" * 80)
        print("TOP PREDICTED NEXT GENES")
        print("=" * 80)
        
        next_predictions = get_next_token_probabilities(model, tokenizer, prompt, args.top_k)
        
        print(f"\n{'Rank':<6} {'Gene':<30} {'Probability':<15} {'Percentage':<10}")
        print("-" * 80)
        for i, pred in enumerate(next_predictions, 1):
            print(f"{i:<6} {pred['gene']:<30} {pred['probability']:<15.6f} {pred['percentage']:<10.2f}%")
        
        df = pd.DataFrame(next_predictions)
        df.to_csv('next_gene_predictions.csv', index=False)
        print(f"\n✓ Saved predictions to: next_gene_predictions.csv")
    
    if args.mode in ['preceding', 'both'] and args.target_gene:
        print("\n" + "=" * 80)
        print(f"GENES MOST LIKELY TO PRECEDE: {args.target_gene}")
        print("=" * 80)
        
        context_genes = prompt.split()
        
        preceding_predictions = find_preceding_genes(
            model, tokenizer, args.target_gene, context_genes, args.top_k
        )
        
        print(f"\n{'Rank':<6} {'Preceding Gene':<30} {'Probability':<15} {'Percentage':<10}")
        print("-" * 80)
        for i, pred in enumerate(preceding_predictions, 1):
            print(f"{i:<6} {pred['preceding_gene']:<30} {pred['probability']:<15.6f} {pred['percentage']:<10.2f}%")
        
        df = pd.DataFrame(preceding_predictions)
        df.to_csv(f'preceding_genes_for_{args.target_gene}.csv', index=False)
        print(f"\n✓ Saved predictions to: preceding_genes_for_{args.target_gene}.csv")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
