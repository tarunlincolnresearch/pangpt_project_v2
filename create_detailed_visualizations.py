import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Data from logs
results = {
    'win128': {
        'final_train_loss': 0.908,
        'final_val_loss': 1.043,
        'final_perplexity': 2.84,
        'final_accuracy': 0.586,
        'final_f1': 0.488,
        'epochs_trained': 115,
        'window_size': 128
    },
    'win256': {
        'final_train_loss': 0.771,
        'final_val_loss': 0.912,
        'final_perplexity': 2.49,
        'final_accuracy': 0.655,
        'final_f1': 0.587,
        'epochs_trained': 82,
        'window_size': 256
    },
    'win512': {
        'final_train_loss': 0.700,
        'final_val_loss': 1.036,
        'final_perplexity': 2.82,
        'final_accuracy': 0.697,
        'final_f1': 0.546,
        'epochs_trained': 47,
        'window_size': 512
    },
    'win1024': {
        'final_train_loss': 0.687,
        'final_val_loss': 1.405,
        'final_perplexity': 4.08,
        'final_accuracy': 0.599,
        'final_f1': 0.491,
        'epochs_trained': 44,
        'window_size': 1024
    },
    'win2048': {
        'final_train_loss': 0.720,
        'final_val_loss': 1.111,
        'final_perplexity': 3.04,
        'final_accuracy': 0.651,
        'final_f1': 0.557,
        'epochs_trained': 54,
        'window_size': 2048
    }
}

output_dir = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/results/analysis"
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(results).T
df = df.sort_values('window_size')

# Define colors for each window
colors = {
    'win128': '#FF6B6B',
    'win256': '#4ECDC4',  # Best model - teal
    'win512': '#45B7D1',
    'win1024': '#FFA07A',
    'win2048': '#98D8C8'
}

markers = {
    'win128': 'o',
    'win256': 's',  # Square for best
    'win512': '^',
    'win1024': 'D',
    'win2048': 'v'
}

# ============================================================================
# FIGURE 1: Line Graphs - All Metrics Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Window Size Performance Comparison - All Metrics', 
             fontsize=18, fontweight='bold', y=0.995)

window_sizes = df['window_size'].values

# Plot 1: Validation Loss
ax = axes[0, 0]
for idx, row in df.iterrows():
    ax.plot(row['window_size'], row['final_val_loss'], 
            marker=markers[idx], markersize=15, linewidth=0,
            color=colors[idx], label=idx.upper(), alpha=0.8)
ax.plot(window_sizes, df['final_val_loss'].values, 
        'k--', alpha=0.3, linewidth=2, zorder=0)
ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax.set_title('Validation Loss vs Window Size\n(Lower is Better)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.set_xscale('log', base=2)
ax.set_xticks(window_sizes)
ax.set_xticklabels(window_sizes)
# Highlight best
best_val = df['final_val_loss'].min()
best_win = df[df['final_val_loss'] == best_val]['window_size'].values[0]
ax.axhline(y=best_val, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax.text(window_sizes[-1]*1.1, best_val, f'Best: {best_val:.3f}', 
        fontsize=10, color='green', fontweight='bold')

# Plot 2: Perplexity
ax = axes[0, 1]
for idx, row in df.iterrows():
    ax.plot(row['window_size'], row['final_perplexity'], 
            marker=markers[idx], markersize=15, linewidth=0,
            color=colors[idx], label=idx.upper(), alpha=0.8)
ax.plot(window_sizes, df['final_perplexity'].values, 
        'k--', alpha=0.3, linewidth=2, zorder=0)
ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
ax.set_title('Perplexity vs Window Size\n(Lower is Better)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.set_xscale('log', base=2)
ax.set_xticks(window_sizes)
ax.set_xticklabels(window_sizes)
best_perp = df['final_perplexity'].min()
ax.axhline(y=best_perp, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax.text(window_sizes[-1]*1.1, best_perp, f'Best: {best_perp:.2f}', 
        fontsize=10, color='green', fontweight='bold')

# Plot 3: Accuracy
ax = axes[1, 0]
for idx, row in df.iterrows():
    ax.plot(row['window_size'], row['final_accuracy'], 
            marker=markers[idx], markersize=15, linewidth=0,
            color=colors[idx], label=idx.upper(), alpha=0.8)
ax.plot(window_sizes, df['final_accuracy'].values, 
        'k--', alpha=0.3, linewidth=2, zorder=0)
ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs Window Size\n(Higher is Better)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.set_xscale('log', base=2)
ax.set_xticks(window_sizes)
ax.set_xticklabels(window_sizes)
best_acc = df['final_accuracy'].max()
ax.axhline(y=best_acc, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax.text(window_sizes[-1]*1.1, best_acc, f'Best: {best_acc:.3f}', 
        fontsize=10, color='green', fontweight='bold')

# Plot 4: F1 Score
ax = axes[1, 1]
for idx, row in df.iterrows():
    ax.plot(row['window_size'], row['final_f1'], 
            marker=markers[idx], markersize=15, linewidth=0,
            color=colors[idx], label=idx.upper(), alpha=0.8)
ax.plot(window_sizes, df['final_f1'].values, 
        'k--', alpha=0.3, linewidth=2, zorder=0)
ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('F1 Score vs Window Size\n(Higher is Better)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.set_xscale('log', base=2)
ax.set_xticks(window_sizes)
ax.set_xticklabels(window_sizes)
best_f1 = df['final_f1'].max()
ax.axhline(y=best_f1, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax.text(window_sizes[-1]*1.1, best_f1, f'Best: {best_f1:.3f}', 
        fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/line_graphs_all_metrics.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/line_graphs_all_metrics.png")

# ============================================================================
# FIGURE 2: Individual Metric - Validation Loss (Detailed)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
for idx, row in df.iterrows():
    linewidth = 4 if idx == 'win256' else 2
    alpha = 1.0 if idx == 'win256' else 0.7
    ax.plot(row['window_size'], row['final_val_loss'], 
            marker=markers[idx], markersize=20 if idx == 'win256' else 15,
            linewidth=linewidth, color=colors[idx], 
            label=f"{idx.upper()} (Loss: {row['final_val_loss']:.3f})", 
            alpha=alpha, zorder=10 if idx == 'win256' else 5)

ax.plot(window_sizes, df['final_val_loss'].values, 
        'k--', alpha=0.2, linewidth=2, zorder=0, label='Trend')

ax.set_xlabel('Window Size (tokens)', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
ax.set_title('Validation Loss Comparison Across Window Sizes\n' + 
             '🏆 WIN256 Achieves Lowest Validation Loss (Best Generalization)',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
ax.set_xscale('log', base=2)
ax.set_xticks(window_sizes)
ax.set_xticklabels(window_sizes)

# Add annotation for best model
best_idx = df['final_val_loss'].idxmin()
best_row = df.loc[best_idx]
ax.annotate('BEST MODEL\nLowest Val Loss', 
            xy=(best_row['window_size'], best_row['final_val_loss']),
            xytext=(best_row['window_size']*0.5, best_row['final_val_loss']+0.15),
            fontsize=12, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))

plt.tight_layout()
plt.savefig(f'{output_dir}/validation_loss_detailed.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/validation_loss_detailed.png")

# ============================================================================
# FIGURE 3: Individual Metric - Perplexity (Detailed)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
for idx, row in df.iterrows():
    linewidth = 4 if idx == 'win256' else 2
    alpha = 1.0 if idx == 'win256' else 0.7
    ax.plot(row['window_size'], row['final_perplexity'], 
            marker=markers[idx], markersize=20 if idx == 'win256' else 15,
            linewidth=linewidth, color=colors[idx], 
            label=f"{idx.upper()} (Perp: {row['final_perplexity']:.2f})", 
            alpha=alpha, zorder=10 if idx == 'win256' else 5)

ax.plot(window_sizes, df['final_perplexity'].values, 
        'k--', alpha=0.2, linewidth=2, zorder=0, label='Trend')

ax.set_xlabel('Window Size (tokens)', fontsize=14, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
ax.set_title('Perplexity Comparison Across Window Sizes\n' + 
             '🏆 WIN256 Achieves Lowest Perplexity (Most Confident Predictions)',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
ax.set_xscale('log', base=2)
ax.set_xticks(window_sizes)
ax.set_xticklabels(window_sizes)

# Add annotation
best_idx = df['final_perplexity'].idxmin()
best_row = df.loc[best_idx]
ax.annotate('BEST MODEL\nLowest Perplexity', 
            xy=(best_row['window_size'], best_row['final_perplexity']),
            xytext=(best_row['window_size']*0.5, best_row['final_perplexity']+0.6),
            fontsize=12, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))

plt.tight_layout()
plt.savefig(f'{output_dir}/perplexity_detailed.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/perplexity_detailed.png")

# ============================================================================
# FIGURE 4: Training vs Validation Loss Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(df))
width = 0.35

bars1 = ax.bar(x - width/2, df['final_train_loss'], width, 
               label='Training Loss', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, df['final_val_loss'], width, 
               label='Validation Loss', alpha=0.8, edgecolor='black', linewidth=1.5)

# Color bars
for i, (idx, bar1, bar2) in enumerate(zip(df.index, bars1, bars2)):
    bar1.set_facecolor(colors[idx])
    bar2.set_facecolor(colors[idx])
    bar2.set_alpha(0.6)
    
    # Highlight best model
    if idx == 'win256':
        bar1.set_edgecolor('darkgreen')
        bar2.set_edgecolor('darkgreen')
        bar1.set_linewidth(3)
        bar2.set_linewidth(3)

ax.set_xlabel('Window Size', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Training vs Validation Loss Comparison\n' +
             'Checking for Overfitting Across Window Sizes',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([idx.upper() for idx in df.index], fontsize=11)
ax.legend(fontsize=12, framealpha=0.95, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/train_vs_val_loss.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/train_vs_val_loss.png")

# ============================================================================
# FIGURE 5: Comprehensive Metrics Radar Chart
# ============================================================================
from math import pi

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Normalize metrics (0-1 scale, invert loss/perplexity)
metrics = ['Accuracy', 'F1 Score', 'Val Loss\n(inverted)', 'Perplexity\n(inverted)', 'Train Loss\n(inverted)']
num_vars = len(metrics)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

for idx, row in df.iterrows():
    values = [
        row['final_accuracy'],
        row['final_f1'],
        1 - (row['final_val_loss'] / df['final_val_loss'].max()),  # Inverted and normalized
        1 - (row['final_perplexity'] / df['final_perplexity'].max()),  # Inverted and normalized
        1 - (row['final_train_loss'] / df['final_train_loss'].max())  # Inverted and normalized
    ]
    values += values[:1]
    
    linewidth = 3 if idx == 'win256' else 1.5
    alpha = 0.3 if idx == 'win256' else 0.15
    
    ax.plot(angles, values, 'o-', linewidth=linewidth, 
            label=idx.upper(), color=colors[idx])
    ax.fill(angles, values, alpha=alpha, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('Comprehensive Performance Comparison\n(All Metrics Normalized)', 
             fontsize=14, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/radar_chart_all_windows.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/radar_chart_all_windows.png")

print("\n" + "="*80)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  1. line_graphs_all_metrics.png - Line graphs for all metrics")
print("  2. validation_loss_detailed.png - Detailed validation loss comparison")
print("  3. perplexity_detailed.png - Detailed perplexity comparison")
print("  4. train_vs_val_loss.png - Training vs validation loss")
print("  5. radar_chart_all_windows.png - Comprehensive radar chart")
print("="*80 + "\n")
