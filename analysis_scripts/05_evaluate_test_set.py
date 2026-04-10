import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

def evaluate_sequence(model, tokenizer, sequence, device):
    """Evaluate model on a single sequence"""
    tokens = tokenizer.encode(sequence).ids
    
    if len(tokens) < 2:
        return None
    
    predictions = []
    actuals = []
    
    # Predict each token given previous context
    for i in range(1, min(len(tokens), 50)):  # Limit to first 50 tokens
        context = tokens[:i]
        actual_next = tokens[i]
        
        input_ids = torch.tensor([context]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[0, -1, :]
            predicted = torch.argmax(logits).item()
        
        predictions.append(predicted)
        actuals.append(actual_next)
    
    return predictions, actuals

print("="*80)
print("TEST SET EVALUATION")
print("="*80)

# Load model
print("\nLoading model...")
model_path = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth"
tokenizer_path = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json"

tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleTransformerModel(vocab_size, 512, 8, 6, 512, dropout_rate=0.1, pe_max_len=5000, pe_dropout_rate=0.1)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load test data
print("Loading test data...")
test_file = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/data/phase1/win128/test_windows.txt"

test_sequences = []
with open(test_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            test_sequences.append(line)

print(f"Loaded {len(test_sequences)} test sequences")

# Evaluate on sample of test set (first 100 sequences)
sample_size = min(100, len(test_sequences))
print(f"\nEvaluating on {sample_size} sequences...")

all_predictions = []
all_actuals = []
sequence_accuracies = []

for i, seq in enumerate(test_sequences[:sample_size]):
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{sample_size} sequences...")
    
    result = evaluate_sequence(model, tokenizer, seq, device)
    if result:
        preds, acts = result
        all_predictions.extend(preds)
        all_actuals.extend(acts)
        
        seq_acc = accuracy_score(acts, preds)
        sequence_accuracies.append(seq_acc)

# Calculate metrics
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

overall_accuracy = accuracy_score(all_actuals, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_actuals, all_predictions, average='weighted', zero_division=0)

print(f"\nOverall Metrics:")
print(f"  • Accuracy:  {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"  • Precision: {precision:.4f}")
print(f"  • Recall:    {recall:.4f}")
print(f"  • F1 Score:  {f1:.4f}")

print(f"\nPer-Sequence Statistics:")
print(f"  • Mean Accuracy:   {np.mean(sequence_accuracies):.4f}")
print(f"  • Median Accuracy: {np.median(sequence_accuracies):.4f}")
print(f"  • Std Dev:         {np.std(sequence_accuracies):.4f}")
print(f"  • Min Accuracy:    {np.min(sequence_accuracies):.4f}")
print(f"  • Max Accuracy:    {np.max(sequence_accuracies):.4f}")

# Analyze prediction distribution
unique_preds = len(set(all_predictions))
unique_actuals = len(set(all_actuals))

print(f"\nPrediction Diversity:")
print(f"  • Unique predicted genes: {unique_preds}")
print(f"  • Unique actual genes:    {unique_actuals}")
print(f"  • Coverage:               {unique_preds/unique_actuals*100:.2f}%")

# Save results
output_dir = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/results/analysis"
os.makedirs(output_dir, exist_ok=True)

results_df = pd.DataFrame({
    'sequence_id': range(len(sequence_accuracies)),
    'accuracy': sequence_accuracies
})
results_df.to_csv(f'{output_dir}/test_set_evaluation.csv', index=False)

# Save summary
summary = {
    'overall_accuracy': overall_accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'mean_sequence_accuracy': np.mean(sequence_accuracies),
    'median_sequence_accuracy': np.median(sequence_accuracies),
    'std_sequence_accuracy': np.std(sequence_accuracies),
    'unique_predictions': unique_preds,
    'unique_actuals': unique_actuals,
    'coverage': unique_preds/unique_actuals
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{output_dir}/test_set_summary.csv', index=False)

print(f"\n✓ Saved results to: {output_dir}/test_set_evaluation.csv")
print(f"✓ Saved summary to: {output_dir}/test_set_summary.csv")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80 + "\n")
