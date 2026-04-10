import re
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_training_metrics(log_file):
    """Extract epoch-by-epoch training and validation metrics from log file"""
    epochs = []
    train_losses = []
    val_losses = []
    perplexities = []
    accuracies = []
    f1_scores = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract training loss per epoch
    train_pattern = r'Epoch (\d+) - Training Loss: ([\d.]+), Perplexity: ([\d.]+)'
    train_matches = re.findall(train_pattern, content)
    
    # Extract validation metrics per epoch
    val_pattern = r'Epoch (\d+) - Validation Loss: ([\d.]+), Perplexity: ([\d.]+), Accuracy: ([\d.]+).*?F1: ([\d.]+)'
    val_matches = re.findall(val_pattern, content)
    
    for epoch, train_loss, train_perp in train_matches:
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
    
    for epoch, val_loss, val_perp, acc, f1 in val_matches:
        val_losses.append(float(val_loss))
        perplexities.append(float(val_perp))
        accuracies.append(float(acc))
        f1_scores.append(float(f1))
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'perplexity': perplexities,
        'accuracy': accuracies,
        'f1_score': f1_scores
    }

# Define log files for each window
log_files = {
    'WIN128': '/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/phase1_0_10052.err',
    'WIN256': '/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/phase1_1_10053.err',
    'WIN512': '/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/phase1_4_10051.err',
    'WIN1024': '/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/phase1_3_10055.err',
    'WIN2048': '/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/phase1_2_10054.err'
}

# Colors for each window
colors = {
    'WIN128': '#FF6B6B',
    'WIN256': '#4ECDC4',  # Best model
    'WIN512': '#45B7D1',
    'WIN1024': '#FFA07A',
    'WIN2048': '#98D8C8'
}

# Extract data for all windows
all_data = {}
print("Extracting training data from logs...")
for window, log_file in log_files.items():
    if os.path.exists(log_file):
        print(f"  Processing {window}...")
        all_data[window] = extract_training_metrics(log_file)
    else:
        print(f"  ⚠ Warning: {log_file} not found")

output_dir = "/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/results/analysis"
os.makedirs(output_dir, exist_ok=True)

# Remove old visualizations
print("\nRemoving old visualizations...")
old_files = [
    'window_comparison_comprehensive.png',
    'line_graphs_all_metrics.png',
    'validation_loss_detailed.png',
    'perplexity_detailed.png',
    'train_vs_val_loss.png',
    'radar_chart_all_windows.png'
]
for old_file in old_files:
    file_path = os.path.join(output_dir, old_file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"  ✓ Removed {old_file}")

print("\nCreating new epoch-by-epoch training curve visualizations...")

# ============================================================================
# FIGURE 1: Training Loss Over Epochs
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

for window, data in all_data.items():
    linewidth = 3 if window == 'WIN256' else 2
    alpha = 1.0 if window == 'WIN256' else 0.7
    ax.plot(data['epochs'], data['train_loss'], 
            label=window, color=colors[window], 
            linewidth=linewidth, alpha=alpha, marker='o', markersize=3, markevery=5)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
ax.set_title('Training Loss Over Epochs - All Window Sizes\n' +
             'Lower is Better | WIN256 Shows Best Convergence',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best', framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/training_loss_over_epochs.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: training_loss_over_epochs.png")

# ============================================================================
# FIGURE 2: Validation Loss Over Epochs
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

for window, data in all_data.items():
    linewidth = 3 if window == 'WIN256' else 2
    alpha = 1.0 if window == 'WIN256' else 0.7
    ax.plot(data['epochs'], data['val_loss'], 
            label=f"{window} (Best: {min(data['val_loss']):.3f})", 
            color=colors[window], linewidth=linewidth, alpha=alpha, 
            marker='s', markersize=3, markevery=5)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
ax.set_title('Validation Loss Over Epochs - All Window Sizes\n' +
             '🏆 WIN256 Achieves Lowest Validation Loss (Best Generalization)',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

# Highlight best model's best epoch
win256_data = all_data['WIN256']
best_val_loss = min(win256_data['val_loss'])
best_epoch = win256_data['epochs'][win256_data['val_loss'].index(best_val_loss)]
ax.plot(best_epoch, best_val_loss, 'g*', markersize=20, 
        label=f'Best: Epoch {best_epoch}', zorder=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/validation_loss_over_epochs.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: validation_loss_over_epochs.png")

# ============================================================================
# FIGURE 3: Perplexity Over Epochs
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

for window, data in all_data.items():
    linewidth = 3 if window == 'WIN256' else 2
    alpha = 1.0 if window == 'WIN256' else 0.7
    ax.plot(data['epochs'], data['perplexity'], 
            label=f"{window} (Best: {min(data['perplexity']):.2f})", 
            color=colors[window], linewidth=linewidth, alpha=alpha,
            marker='^', markersize=3, markevery=5)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
ax.set_title('Perplexity Over Epochs - All Window Sizes\n' +
             '🏆 WIN256 Achieves Lowest Perplexity (Most Confident Predictions)',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/perplexity_over_epochs.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: perplexity_over_epochs.png")

# ============================================================================
# FIGURE 4: Accuracy Over Epochs
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

for window, data in all_data.items():
    linewidth = 3 if window == 'WIN256' else 2
    alpha = 1.0 if window == 'WIN256' else 0.7
    ax.plot(data['epochs'], data['accuracy'], 
            label=f"{window} (Best: {max(data['accuracy']):.3f})", 
            color=colors[window], linewidth=linewidth, alpha=alpha,
            marker='D', markersize=3, markevery=5)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Accuracy Over Epochs - All Window Sizes\n' +
             'Higher is Better | WIN512 Has Highest Peak Accuracy',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_over_epochs.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: accuracy_over_epochs.png")

# ============================================================================
# FIGURE 5: F1 Score Over Epochs
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

for window, data in all_data.items():
    linewidth = 3 if window == 'WIN256' else 2
    alpha = 1.0 if window == 'WIN256' else 0.7
    ax.plot(data['epochs'], data['f1_score'], 
            label=f"{window} (Best: {max(data['f1_score']):.3f})", 
            color=colors[window], linewidth=linewidth, alpha=alpha,
            marker='v', markersize=3, markevery=5)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
ax.set_title('F1 Score Over Epochs - All Window Sizes\n' +
             'Higher is Better | WIN256 Achieves Best F1 Score',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/f1_score_over_epochs.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: f1_score_over_epochs.png")

# ============================================================================
# FIGURE 6: Combined - Training vs Validation Loss (All Windows)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Training vs Validation Loss Over Epochs - Individual Window Analysis', 
             fontsize=16, fontweight='bold')

for idx, (window, data) in enumerate(all_data.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    ax.plot(data['epochs'], data['train_loss'], 
            label='Training Loss', color=colors[window], 
            linewidth=2.5, alpha=0.8, linestyle='-')
    ax.plot(data['epochs'], data['val_loss'], 
            label='Validation Loss', color=colors[window], 
            linewidth=2.5, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title(f'{window}\n(Window Size: {window[3:]} tokens)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight if overfitting
    final_gap = data['val_loss'][-1] - data['train_loss'][-1]
    if final_gap > 0.2:
        ax.text(0.95, 0.95, f'Gap: {final_gap:.3f}\n⚠ Overfitting', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Remove empty subplot if odd number of windows
if len(all_data) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{output_dir}/train_vs_val_all_windows.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: train_vs_val_all_windows.png")

# ============================================================================
# FIGURE 7: All Metrics Combined (2x2 Grid)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Comprehensive Training Metrics Over Epochs - All Window Sizes', 
             fontsize=17, fontweight='bold', y=0.995)

# Validation Loss
ax = axes[0, 0]
for window, data in all_data.items():
    lw = 3 if window == 'WIN256' else 2
    ax.plot(data['epochs'], data['val_loss'], label=window, 
            color=colors[window], linewidth=lw, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax.set_title('Validation Loss (Lower is Better)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Perplexity
ax = axes[0, 1]
for window, data in all_data.items():
    lw = 3 if window == 'WIN256' else 2
    ax.plot(data['epochs'], data['perplexity'], label=window, 
            color=colors[window], linewidth=lw, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
ax.set_title('Perplexity (Lower is Better)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Accuracy
ax = axes[1, 0]
for window, data in all_data.items():
    lw = 3 if window == 'WIN256' else 2
    ax.plot(data['epochs'], data['accuracy'], label=window, 
            color=colors[window], linewidth=lw, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy (Higher is Better)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# F1 Score
ax = axes[1, 1]
for window, data in all_data.items():
    lw = 3 if window == 'WIN256' else 2
    ax.plot(data['epochs'], data['f1_score'], label=window, 
            color=colors[window], linewidth=lw, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('F1 Score (Higher is Better)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/all_metrics_combined.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: all_metrics_combined.png")

print("\n" + "="*80)
print("EPOCH-BY-EPOCH TRAINING CURVES CREATED SUCCESSFULLY!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  1. training_loss_over_epochs.png")
print("  2. validation_loss_over_epochs.png")
print("  3. perplexity_over_epochs.png")
print("  4. accuracy_over_epochs.png")
print("  5. f1_score_over_epochs.png")
print("  6. train_vs_val_all_windows.png")
print("  7. all_metrics_combined.png")
print("="*80 + "\n")
