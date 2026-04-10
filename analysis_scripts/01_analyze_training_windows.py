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
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
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

# Rank by validation loss
ranked_by_val_loss = df.sort_values('final_val_loss')
report.append("1. RANKING BY VALIDATION LOSS (Lower is Better):")
report.append("")
for i, (idx, row) in enumerate(ranked_by_val_loss.iterrows(), 1):
    report.append(f"   {i}. {idx.upper()}: {row['final_val_loss']:.4f}")
report.append("")

# Rank by perplexity
ranked_by_perplexity = df.sort_values('final_perplexity')
report.append("2. RANKING BY PERPLEXITY (Lower is Better):")
report.append("")
for i, (idx, row) in enumerate(ranked_by_perplexity.iterrows(), 1):
    report.append(f"   {i}. {idx.upper()}: {row['final_perplexity']:.4f}")
report.append("")

# Rank by accuracy
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

report_text = "\n".join(report)
print("\n" + report_text)

# Save report
with open(f'{output_dir}/comprehensive_training_report.txt', 'w') as f:
    f.write(report_text)

print(f"\n✓ Report saved: {output_dir}/comprehensive_training_report.txt")
print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}\n")
