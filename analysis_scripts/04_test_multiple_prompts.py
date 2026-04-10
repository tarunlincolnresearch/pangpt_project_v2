import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn
import math
import pandas as pd
import os

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

def analyze_prompt(model, tokenizer, prompt, top_k=10):
    """Analyze a single prompt and return predictions"""
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

# Load model
print("Loading model...")
model_path = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth"
tokenizer_path = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json"

tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

model = SimpleTransformerModel(vocab_size, 512, 8, 6, 512, dropout_rate=0.1, pe_max_len=5000, pe_dropout_rate=0.1)
checkpoint = torch.load(model_path, map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')

# Test prompts
test_prompts = [
    "cbtA_3 group_24448 yeeW_1 group_49138 group_72007",
    "nanS group_17144",
    "cbtA_3 yeeW_1",
    "group_24448 group_49138 group_72007 group_28085",
    "nanS group_17144 group_70557"
]

output_dir = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/results/analysis"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*80)
print("TESTING MULTIPLE PROMPTS")
print("="*80)

all_results = []

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"PROMPT {i}: {prompt}")
    print(f"{'='*80}")
    
    predictions = analyze_prompt(model, tokenizer, prompt, top_k=10)
    
    print(f"\n{'Rank':<6} {'Gene':<30} {'Probability':<15} {'Percentage':<10}")
    print("-"*80)
    for rank, pred in enumerate(predictions, 1):
        print(f"{rank:<6} {pred['gene']:<30} {pred['probability']:<15.6f} {pred['percentage']:<10.2f}%")
    
    # Check for repetition
    prompt_genes = set(prompt.split())
    repeated = [p for p in predictions if p['gene'] in prompt_genes]
    
    if repeated:
        print(f"\n⚠️  WARNING: {len(repeated)} predicted genes already in prompt!")
        for r in repeated:
            print(f"   - {r['gene']} ({r['percentage']:.2f}%)")
    
    # Store results
    for rank, pred in enumerate(predictions, 1):
        all_results.append({
            'prompt_id': i,
            'prompt': prompt,
            'rank': rank,
            'gene': pred['gene'],
            'probability': pred['probability'],
            'percentage': pred['percentage'],
            'is_repetition': pred['gene'] in prompt_genes
        })

# Save all results
df = pd.DataFrame(all_results)
df.to_csv(f'{output_dir}/multiple_prompts_analysis.csv', index=False)
print(f"\n✓ Saved all results to: {output_dir}/multiple_prompts_analysis.csv")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTotal prompts tested: {len(test_prompts)}")
print(f"Total predictions: {len(all_results)}")
print(f"Predictions with repetition: {df['is_repetition'].sum()} ({df['is_repetition'].sum()/len(all_results)*100:.1f}%)")
print(f"Average top-1 probability: {df[df['rank']==1]['probability'].mean():.4f}")
print(f"Average top-1 percentage: {df[df['rank']==1]['percentage'].mean():.2f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
