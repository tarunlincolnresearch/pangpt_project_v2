#!/usr/bin/env python3
"""
Analyze genome sequences: Given N genes, predict next gene and compare with actual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import math
import pandas as pd
import numpy as np

# ============================================================
# Model Architecture
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

def predict_next_gene_with_probabilities(model, tokenizer, prompt, device, top_k=30):
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
    
    return results, probabilities

def analyze_genome_sequence(genome_id, genes, model, tokenizer, device, prompt_length=10, top_k=30):
    """
    Analyze a genome sequence:
    - Take first N genes as prompt
    - Predict next gene
    - Compare with actual next gene
    - Show probability distribution
    """
    
    # Split into prompt and actual next gene
    prompt_genes = genes[:prompt_length]
    actual_next_gene = genes[prompt_length] if len(genes) > prompt_length else None
    
    prompt = ' '.join(prompt_genes)
    
    # Get predictions
    predictions, all_probs = predict_next_gene_with_probabilities(model, tokenizer, prompt, device, top_k)
    
    # Find rank of actual gene in predictions
    actual_gene_rank = None
    actual_gene_prob = 0.0
    
    if actual_next_gene:
        for rank, pred in enumerate(predictions, 1):
            if pred['gene'] == actual_next_gene:
                actual_gene_rank = rank
                actual_gene_prob = pred['percentage']
                break
        
        # If not in top-k, find its probability
        if actual_gene_rank is None:
            try:
                actual_token_id = tokenizer.token_to_id(actual_next_gene)
                if actual_token_id is not None:
                    actual_gene_prob = float(all_probs[actual_token_id].cpu().numpy()) * 100
                    actual_gene_rank = f">30 ({actual_gene_prob:.4f}%)"
            except:
                actual_gene_rank = "Not in vocab"
                actual_gene_prob = 0.0
    
    result = {
        'genome_id': genome_id,
        'prompt_genes': prompt_genes,
        'actual_next_gene': actual_next_gene,
        'actual_gene_rank': actual_gene_rank,
        'actual_gene_prob': actual_gene_prob,
        'top_predictions': predictions,
        'top_1_gene': predictions[0]['gene'],
        'top_1_prob': predictions[0]['percentage'],
        'is_correct': predictions[0]['gene'] == actual_next_gene if actual_next_gene else False
    }
    
    return result

def save_and_display_results(results, output_dir):
    """Save results and create formatted display"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed report
    with open(f"{output_dir}/genome_predictions_report.txt", 'w') as f:
        f.write("="*120 + "\n")
        f.write("PHASE 2A MODEL - GENOME NEXT GENE PREDICTION ANALYSIS\n")
        f.write("="*120 + "\n\n")
        
        for r in results:
            f.write(f"\n{'='*120}\n")
            f.write(f"GENOME {r['genome_id']}\n")
            f.write(f"{'='*120}\n\n")
            
            # Show prompt
            f.write(f"Input Genes (Prompt):\n")
            f.write(f"  {' → '.join(r['prompt_genes'])}\n\n")
            
            # Show actual next gene
            f.write(f"ACTUAL NEXT GENE: {r['actual_next_gene']}\n")
            if r['actual_gene_rank']:
                f.write(f"  Model's rank for actual gene: {r['actual_gene_rank']}\n")
                f.write(f"  Probability: {r['actual_gene_prob']:.2f}%\n")
            f.write(f"\n")
            
            # Show top prediction
            f.write(f"MODEL'S TOP PREDICTION: {r['top_1_gene']} ({r['top_1_prob']:.2f}%)\n")
            if r['is_correct']:
                f.write(f"  ✅ CORRECT!\n")
            else:
                f.write(f"  ❌ INCORRECT\n")
            f.write(f"\n")
            
            # Show probability table
            f.write(f"TOP 30 PREDICTIONS (Descending Probability):\n")
            f.write(f"{'Rank':<6} {'Gene':<25} {'Probability':<15} {'Percentage':<12} {'Status':<15}\n")
            f.write(f"{'-'*100}\n")
            
            for i, pred in enumerate(r['top_predictions'], 1):
                status = ""
                if pred['gene'] == r['actual_next_gene']:
                    status = "← ACTUAL GENE"
                f.write(f"{i:<6} {pred['gene']:<25} {pred['probability']:<15.6f} {pred['percentage']:<12.2f}% {status:<15}\n")
            
            f.write("\n")
    
    print(f"✅ Saved: {output_dir}/genome_predictions_report.txt")
    
    # Create CSV summary
    summary_data = []
    for r in results:
        summary_data.append({
            'Genome_ID': r['genome_id'],
            'Actual_Next_Gene': r['actual_next_gene'],
            'Predicted_Gene': r['top_1_gene'],
            'Correct': '✅' if r['is_correct'] else '❌',
            'Top_1_Prob_%': f"{r['top_1_prob']:.2f}",
            'Actual_Gene_Rank': r['actual_gene_rank'],
            'Actual_Gene_Prob_%': f"{r['actual_gene_prob']:.2f}" if r['actual_gene_prob'] > 0 else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    print(f"✅ Saved: {output_dir}/summary.csv")
    
    # Create detailed predictions CSV
    detailed_data = []
    for r in results:
        for rank, pred in enumerate(r['top_predictions'], 1):
            is_actual = "YES" if pred['gene'] == r['actual_next_gene'] else "NO"
            detailed_data.append({
                'Genome_ID': r['genome_id'],
                'Rank': rank,
                'Gene': pred['gene'],
                'Probability': pred['probability'],
                'Percentage': pred['percentage'],
                'Is_Actual_Gene': is_actual
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(f"{output_dir}/detailed_predictions.csv", index=False)
    print(f"✅ Saved: {output_dir}/detailed_predictions.csv")
    
    # Print summary
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}\n")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*120}\n")
    
    # Print accuracy
    correct_count = sum(1 for r in results if r['is_correct'])
    print(f"Top-1 Accuracy: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"\n")

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
    
    # Load test data
    print("Loading test data...")
    with open("../data/phase1/win256/test_windows.txt", 'r') as f:
        test_data = [line.strip().split() for line in f]
    
    # Select 5 diverse genome sequences
    indices = [0, len(test_data)//4, len(test_data)//2, 3*len(test_data)//4, len(test_data)-1]
    
    print(f"Analyzing 5 genome sequences...\n")
    
    results = []
    for i, idx in enumerate(indices, 1):
        print(f"Analyzing genome {i}/5...")
        genes = test_data[idx]
        result = analyze_genome_sequence(i, genes, model, tokenizer, device, prompt_length=10, top_k=30)
        results.append(result)
    
    # Save and display results
    save_and_display_results(results, "results/genome_predictions")
    
    print("✅ Analysis complete!")
    print("Results saved in: results/genome_predictions/")

