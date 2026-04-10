#!/usr/bin/env python3
"""
Analyze multiple prompts and save results to CSV and detailed report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import math
import pandas as pd
import numpy as np

# ============================================================
# Model Architecture (same as before)
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
# Functions
# ============================================================

def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    model = SimpleTransformerModel(
        vocab_size=vocab_size, embed_dim=256, num_heads=8, num_layers=6,
        max_seq_length=512, dropout_rate=0.15, pe_max_len=5000, pe_dropout_rate=0.1
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def predict_next_gene_with_probabilities(model, tokenizer, prompt, device, top_k=20):
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        last_token_logits = outputs[0, -1, :]
        probabilities = F.softmax(last_token_logits, dim=0)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            gene = tokenizer.id_to_token(int(idx))
            results.append({
                'gene': gene,
                'probability': float(prob),
                'percentage': float(prob) * 100
            })
    
    return results

def analyze_multiple_prompts(model, tokenizer, prompts, device, top_k=20):
    """Analyze multiple prompts and return comprehensive results"""
    
    all_results = []
    
    for prompt_id, prompt in enumerate(prompts, 1):
        print(f"\nAnalyzing prompt {prompt_id}/{len(prompts)}...")
        
        # Get predictions
        predictions = predict_next_gene_with_probabilities(model, tokenizer, prompt, device, top_k)
        
        # Calculate statistics
        top_1_prob = predictions[0]['percentage']
        top_5_prob = sum(p['percentage'] for p in predictions[:5])
        top_10_prob = sum(p['percentage'] for p in predictions[:10])
        
        # Entropy
        probs = np.array([p['probability'] for p in predictions])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(predictions))
        normalized_entropy = entropy / max_entropy
        
        # Store results
        prompt_result = {
            'prompt_id': prompt_id,
            'prompt': prompt,
            'last_gene': prompt.split()[-1],
            'top_1_gene': predictions[0]['gene'],
            'top_1_prob': top_1_prob,
            'top_5_prob': top_5_prob,
            'top_10_prob': top_10_prob,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'predictions': predictions
        }
        
        all_results.append(prompt_result)
    
    return all_results

def save_results(results, output_dir):
    """Save results to CSV and detailed text report"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary CSV
    summary_data = []
    for r in results:
        summary_data.append({
            'Prompt_ID': r['prompt_id'],
            'Last_Gene': r['last_gene'],
            'Top_1_Prediction': r['top_1_gene'],
            'Top_1_Probability_%': f"{r['top_1_prob']:.2f}",
            'Top_5_Cumulative_%': f"{r['top_5_prob']:.2f}",
            'Top_10_Cumulative_%': f"{r['top_10_prob']:.2f}",
            'Entropy': f"{r['entropy']:.2f}",
            'Confidence': 'High' if r['normalized_entropy'] < 0.4 else 'Medium' if r['normalized_entropy'] < 0.7 else 'Low'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    print(f"✅ Saved: {output_dir}/summary.csv")
    
    # 2. Detailed predictions CSV
    detailed_data = []
    for r in results:
        for rank, pred in enumerate(r['predictions'], 1):
            detailed_data.append({
                'Prompt_ID': r['prompt_id'],
                'Rank': rank,
                'Gene': pred['gene'],
                'Probability': pred['probability'],
                'Percentage': pred['percentage']
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(f"{output_dir}/detailed_predictions.csv", index=False)
    print(f"✅ Saved: {output_dir}/detailed_predictions.csv")
    
    # 3. Detailed text report
    with open(f"{output_dir}/detailed_report.txt", 'w') as f:
        f.write("="*100 + "\n")
        f.write("PHASE 2A MODEL - NEXT GENE PREDICTION ANALYSIS\n")
        f.write("="*100 + "\n\n")
        
        for r in results:
            f.write(f"\n{'='*100}\n")
            f.write(f"PROMPT {r['prompt_id']}\n")
            f.write(f"{'='*100}\n\n")
            
            f.write(f"Input Genes (last 10):\n")
            genes = r['prompt'].split()
            f.write(f"  {' → '.join(genes[-10:])}\n\n")
            
            f.write(f"Statistics:\n")
            f.write(f"  Top 1 probability:  {r['top_1_prob']:>6.2f}%\n")
            f.write(f"  Top 5 probability:  {r['top_5_prob']:>6.2f}%\n")
            f.write(f"  Top 10 probability: {r['top_10_prob']:>6.2f}%\n")
            f.write(f"  Entropy: {r['entropy']:.2f} bits\n")
            f.write(f"  Normalized entropy: {r['normalized_entropy']:.2f}\n\n")
            
            f.write(f"Top 20 Next Gene Predictions:\n")
            f.write(f"{'Rank':<6} {'Gene':<25} {'Probability':<15} {'Percentage':<12}\n")
            f.write(f"{'-'*70}\n")
            
            for i, pred in enumerate(r['predictions'][:20], 1):
                f.write(f"{i:<6} {pred['gene']:<25} {pred['probability']:<15.6f} {pred['percentage']:<12.2f}%\n")
            
            f.write("\n")
    
    print(f"✅ Saved: {output_dir}/detailed_report.txt")
    
    # 4. Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}\n")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*100}\n")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        "checkpoints/model_checkpoint.pth",
        "../pangenome_gpt_tokenizer.json",
        device
    )
    print("✅ Model loaded\n")
    
    # Load test data and get 5 diverse prompts
    print("Loading test data...")
    with open("../data/phase1/win256/test_windows.txt", 'r') as f:
        test_data = [line.strip() for line in f]
    
    # Select 5 diverse prompts (evenly spaced through test set)
    indices = [0, len(test_data)//4, len(test_data)//2, 3*len(test_data)//4, len(test_data)-1]
    prompts = []
    for idx in indices:
        # Take first 10 genes as prompt
        genes = test_data[idx].split()[:10]
        prompts.append(' '.join(genes))
    
    print(f"Selected {len(prompts)} test prompts\n")
    
    # Analyze prompts
    results = analyze_multiple_prompts(model, tokenizer, prompts, device, top_k=30)
    
    # Save results
    save_results(results, "results/next_gene_predictions")
    
    print("\n✅ Analysis complete!")
    print("Results saved in: results/next_gene_predictions/")

