import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Manual data extraction from logs
results = {
    'win128': {
        'final_train_loss': 0.908,
        'final_val_loss': 1.043,
        'final_perplexity': 2.84,
        'best_val_loss': 1.043,
        'best_perplexity': 2.84,
        'final_accuracy': 0.586,
        'final_f1': 0.488,
        'epochs_trained': 115,
        'max_seq_length': 128
    },
    'win256': {
        'final_train_loss': 0.771,
        'final_val_loss': 0.912,
        'final_perplexity': 2.49,
        'best_val_loss': 0.912,
        'best_perplexity': 2.49,
        'final_accuracy': 0.655,
        'final_f1': 0.587,
        'epochs_trained': 82,
        'max_seq_length': 256
    },
    'win512': {
        'final_train_loss': 0.700,
        'final_val_loss': 1.036,
        'final_perplexity': 2.82,
        'best_val_loss': 1.026,
        'best_perplexity': 2.79,
        'final_accuracy': 0.697,
        'final_f1': 0.546,
        'epochs_trained': 47,
        'max_seq_length': 512
    },
    'win1024': {
        'final_train_loss': 0.687,
        'final_val_loss': 1.405,
        'final_perplexity': 4.08,
        'best_val_loss': 1.405,
        'best_perplexity': 4.08,
        'final_accuracy': 0.599,
        'final_f1': 0.491,
        'epochs_trained': 44,
        'max_seq_length': 1024
    },
    'win2048': {
        'final_train_loss': 0.720,
        'final_val_loss': 1.111,
        'final_perplexity': 3.04,
        'best_val_loss': 1.111,
        'best_perplexity': 3.04,
        'final_accuracy': 0.651,
        'final_f1': 0.557,
        'epochs_trained': 54,
        'max_seq_length': 2048
    }
}

# Create output directory
output_dir = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/results/analysis"
os.makedirs(output_dir, exist_ok=True)

# Convert to DataFrame
df = pd.DataFrame(results).T
df['window_size'] = df['max_seq_length']
df = df.sort_values('window_size')

print("="*80)
print("WINDOW SIZE COMPARISON - PHASE 1 TRAINING")
print("="*80)
print("\n", df.to_string())

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Window Size Performance Comparison - Phase 1 Training', fontsize=16, fontweight='bold')

# Plot 1: Validation Loss
ax = axes[0, 0]
bars = ax.bar(df['window_size'], df['final_val_loss'], color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Window Size', fontweight='bold')
ax.set_ylabel('Validation Loss', fontweight='bold')
ax.set_title('Final Validation Loss by Window Size')
ax.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
# Highlight best
best_idx = df['final_val_loss'].idxmin()
bars[list(df.index).index(best_idx)].set_color('green')
bars[list(df.index).index(best_idx)].set_alpha(0.9)

# Plot 2: Perplexity
ax = axes[0, 1]
bars = ax.bar(df['window_size'], df['final_perplexity'], color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Window Size', fontweight='bold')
ax.set_ylabel('Perplexity', fontweight='bold')
ax.set_title('Final Perplexity by Window Size')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
best_idx = df['final_perplexity'].idxmin()
bars[list(df.index).index(best_idx)].set_color('green')
bars[list(df.index).index(best_idx)].set_alpha(0.9)

# Plot 3: Accuracy
ax = axes[0, 2]
bars = ax.bar(df['window_size'], df['final_accuracy'], color='lightgreen', alpha=0.7, edgecolor='black')
ax.set_xlabel('Window Size', fontweight='bold')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Final Accuracy by Window Size')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.5, 0.75])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
best_idx = df['final_accuracy'].idxmax()
bars[list(df.index).index(best_idx)].set_color('darkgreen')
bars[list(df.index).index(best_idx)].set_alpha(0.9)

# Plot 4: F1 Score
ax = axes[1, 0]
bars = ax.bar(df['window_size'], df['final_f1'], color='gold', alpha=0.7, edgecolor='black')
ax.set_xlabel('Window Size', fontweight='bold')
ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('Final F1 Score by Window Size')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
best_idx = df['final_f1'].idxmax()
bars[list(df.index).index(best_idx)].set_color('darkgoldenrod')
bars[list(df.index).index(best_idx)].set_alpha(0.9)

# Plot 5: Training vs Validation Loss
ax = axes[1, 1]
x = np.arange(len(df))
width = 0.35
bars1 = ax.bar(x - width/2, df['final_train_loss'], width, label='Training Loss', 
               color='skyblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, df['final_val_loss'], width, label='Validation Loss', 
               color='salmon', alpha=0.8, edgecolor='black')
ax.set_xlabel('Window Size', fontweight='bold')
ax.set_ylabel('Loss', fontweight='bold')
ax.set_title('Training vs Validation Loss')
ax.set_xticks(x)
ax.set_xticklabels(df['window_size'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 6: Epochs Trained
ax = axes[1, 2]
bars = ax.bar(df['window_size'], df['epochs_trained'], color='mediumpurple', alpha=0.7, edgecolor='black')
ax.set_xlabel('Window Size', fontweight='bold')
ax.set_ylabel('Epochs', fontweight='bold')
ax.set_title('Number of Epochs Trained')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/window_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir}/window_comparison_comprehensive.png")

# Generate detailed report
report = []
report.append("="*80)
report.append("PHASE 1 TRAINING - COMPREHENSIVE ANALYSIS REPORT")
report.append("="*80)
report.append("")
report.append("WINDOW SIZE PERFORMANCE RANKING")
report.append("-"*80)
report.append("")

# Rank by validation loss (lower is better)
ranked_by_val_loss = df.sort_values('final_val_loss')
report.append("1. RANKING BY VALIDATION LOSS (Lower is Better):")
report.append("")
for i, (idx, row) in enumerate(ranked_by_val_loss.iterrows(), 1):
    report.append(f"   {i}. {idx.upper()}: {row['final_val_loss']:.4f}")
report.append("")

# Rank by perplexity (lower is better)
ranked_by_perplexity = df.sort_values('final_perplexity')
report.append("2. RANKING BY PERPLEXITY (Lower is Better):")
report.append("")
for i, (idx, row) in enumerate(ranked_by_perplexity.iterrows(), 1):
    report.append(f"   {i}. {idx.upper()}: {row['final_perplexity']:.4f}")
report.append("")

# Rank by accuracy (higher is better)
ranked_by_accuracy = df.sort_values('final_accuracy', ascending=False)
report.append("3. RANKING BY ACCURACY (Higher is Better):")
report.append("")
for i, (idx, row) in enumerate(ranked_by_accuracy.iterrows(), 1):
    report.append(f"   {i}. {idx.upper()}: {row['final_accuracy']:.4f}")
report.append("")

report.append("="*80)
report.append("BEST MODEL SELECTION")
report.append("="*80)
report.append("")

best_window = ranked_by_val_loss.index[0]
best_data = ranked_by_val_loss.iloc[0]

report.append(f"🏆 SELECTED WINDOW SIZE: {best_window.upper()}")
report.append("")
report.append("Performance Metrics:")
report.append(f"  • Validation Loss:    {best_data['final_val_loss']:.4f}")
report.append(f"  • Perplexity:         {best_data['final_perplexity']:.4f}")
report.append(f"  • Accuracy:           {best_data['final_accuracy']:.4f}")
report.append(f"  • F1 Score:           {best_data['final_f1']:.4f}")
report.append(f"  • Epochs Trained:     {int(best_data['epochs_trained'])}")
report.append("")

report.append("="*80)
report.append("WHY WIN256 WAS SELECTED")
report.append("="*80)
report.append("")
report.append("1. LOWEST VALIDATION LOSS (0.912)")
report.append("   - Better generalization than other window sizes")
report.append("   - Indicates the model learned patterns without overfitting")
report.append("")
report.append("2. LOWEST PERPLEXITY (2.49)")
report.append("   - Model is most confident in its predictions")
report.append("   - Better at predicting next gene in sequence")
report.append("")
report.append("3. GOOD BALANCE OF METRICS")
report.append("   - High accuracy (0.655)")
report.append("   - Best F1 score (0.587)")
report.append("   - Trained efficiently (82 epochs)")
report.append("")
report.append("4. OPTIMAL CONTEXT LENGTH")
report.append("   - 256 tokens provides enough context for gene relationships")
report.append("   - Not too short (like 128) - captures longer dependencies")
report.append("   - Not too long (like 1024/2048) - avoids overfitting to noise")
report.append("")

report.append("="*80)
report.append("COMPARISON WITH OTHER WINDOWS")
report.append("="*80)
report.append("")

report.append("WIN128 (Shortest Context):")
report.append("  ✗ Higher validation loss (1.043)")
report.append("  ✗ Higher perplexity (2.84)")
report.append("  ✗ Limited context - can't capture longer gene dependencies")
report.append("")

report.append("WIN512:")
report.append("  ✗ Higher validation loss (1.036)")
report.append("  ✗ Lower F1 score (0.546)")
report.append("  ~ Good accuracy but worse generalization")
report.append("")

report.append("WIN1024:")
report.append("  ✗ WORST perplexity (4.08)")
report.append("  ✗ Highest validation loss (1.405)")
report.append("  ✗ Overfitting - too much context causes noise")
report.append("")

report.append("WIN2048 (Longest Context):")
report.append("  ✗ Higher validation loss (1.111)")
report.append("  ✗ Higher perplexity (3.04)")
report.append("  ✗ Computational overhead without performance gain")
report.append("")

report.append("="*80)
report.append("TRAINING IMPROVEMENTS TO CONSIDER")
report.append("="*80)
report.append("")
report.append("1. DATA AUGMENTATION")
report.append("   • Add noise/mutations to gene sequences")
report.append("   • Reverse complement sequences")
report.append("   • Random gene order permutations")
report.append("")
report.append("2. ARCHITECTURE IMPROVEMENTS")
report.append("   • Try deeper models (more layers)")
report.append("   • Experiment with different attention mechanisms")
report.append("   • Add residual connections")
report.append("   • Use layer normalization")
report.append("")
report.append("3. TRAINING STRATEGIES")
report.append("   • Curriculum learning (start with easier examples)")
report.append("   • Cosine annealing learning rate schedule")
report.append("   • Gradient clipping to prevent exploding gradients")
report.append("   • Mixed precision training for faster convergence")
report.append("")
report.append("4. REGULARIZATION")
report.append("   • Increase dropout rate")
report.append("   • Add label smoothing")
report.append("   • Use weight decay more aggressively")
report.append("")
report.append("5. DATA PREPROCESSING")
report.append("   • Better tokenization strategies")
report.append("   • Filter low-frequency genes")
report.append("   • Balance dataset if class imbalance exists")
report.append("")

report.append("="*80)
report.append("NEXT STEPS")
report.append("="*80)
report.append("")
report.append("✓ WIN256 selected as best model")
report.append("✓ Model checkpoint saved for inference")
report.append("→ Use panPrompt_v3.py for predictions with optimal sampling")
report.append("→ Consider retraining WIN256 with improvements listed above")
report.append("")

report_text = "\n".join(report)
print("\n" + report_text)

# Save report
with open(f'{output_dir}/comprehensive_training_report.txt', 'w') as f:
    f.write(report_text)

print(f"\n✓ Report saved: {output_dir}/comprehensive_training_report.txt")
print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}\n")
