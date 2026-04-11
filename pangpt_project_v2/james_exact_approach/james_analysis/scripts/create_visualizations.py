import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def create_comparison_visualizations(james_metrics_file=None):
    """
    Create visualizations comparing all phases
    """
    
    phases = ['Phase 1', 'Phase 2', 'Phase 2A', 'Phase 3', 'James']
    
    # Default data
    val_accs = [65.5, 50.28, 50.27, 50.29, None]
    test_accs = [0.51, 50.23, 50.27, 50.29, None]
    perplexities = [2.49, 258.32, 142.32, 141.30, None]
    params = [50, 30, 30, 89, None]
    training_times = [6, 6, 6, 27, None]
    
    # Load James data if available
    if james_metrics_file:
        try:
            with open(james_metrics_file, 'r') as f:
                james_metrics = json.load(f)
            
            val_accs[-1] = james_metrics['validation'].get('final_accuracy')
            test_accs[-1] = james_metrics['test'].get('accuracy')
            perplexities[-1] = james_metrics['test'].get('perplexity')
            
        except Exception as e:
            print(f"Warning: Could not load James metrics: {e}")
    
    # Calculate overfitting gaps
    overfitting_gaps = []
    for val, test in zip(val_accs, test_accs):
        if val is not None and test is not None:
            overfitting_gaps.append(val - test)
        else:
            overfitting_gaps.append(None)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('James Model - Comprehensive Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Validation vs Test Accuracy
    x = np.arange(len(phases))
    width = 0.35
    
    val_plot = [v if v is not None else 0 for v in val_accs]
    test_plot = [t if t is not None else 0 for t in test_accs]
    
    axes[0, 0].bar(x - width/2, val_plot, width, label='Validation', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, test_plot, width, label='Test', alpha=0.8, color='coral')
    axes[0, 0].set_xlabel('Phase', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 0].set_title('Validation vs Test Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(phases, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% plateau')
    
    # 2. Test Perplexity Comparison
    perp_plot = [p if p is not None else 0 for p in perplexities]
    colors_perp = ['red', 'orange', 'yellow', 'yellowgreen', 'gray']
    
    bars = axes[0, 1].bar(phases, perp_plot, alpha=0.8, color=colors_perp)
    axes[0, 1].set_xlabel('Phase', fontweight='bold')
    axes[0, 1].set_ylabel('Perplexity (log scale)', fontweight='bold')
    axes[0, 1].set_title('Test Perplexity Comparison')
    axes[0, 1].set_xticks(range(len(phases)))
    axes[0, 1].set_xticklabels(phases, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_yscale('log')
    
    # 3. Overfitting Gap
    gap_plot = [g if g is not None else 0 for g in overfitting_gaps]
    colors_gap = []
    for gap in overfitting_gaps:
        if gap is None:
            colors_gap.append('gray')
        elif gap > 10:
            colors_gap.append('red')
        elif gap > 5:
            colors_gap.append('orange')
        elif gap > 1:
            colors_gap.append('yellow')
        else:
            colors_gap.append('green')
    
    bars = axes[0, 2].bar(phases, gap_plot, alpha=0.8, color=colors_gap)
    axes[0, 2].set_xlabel('Phase', fontweight='bold')
    axes[0, 2].set_ylabel('Accuracy Gap (%)', fontweight='bold')
    axes[0, 2].set_title('Overfitting Gap (Val - Test)')
    axes[0, 2].set_xticks(range(len(phases)))
    axes[0, 2].set_xticklabels(phases, rotation=45, ha='right')
    axes[0, 2].axhline(y=5, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Overfit threshold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Model Size vs Performance
    params_clean = [(p, t) for p, t in zip(params, test_accs) if p is not None and t is not None]
    if params_clean:
        p_vals, t_vals = zip(*params_clean)
        axes[1, 0].scatter(p_vals, t_vals, s=200, alpha=0.6, c=range(len(p_vals)), cmap='viridis')
        
        for i, (p, t) in enumerate(params_clean):
            phase_idx = [j for j, x in enumerate(params) if x == p][0]
            axes[1, 0].annotate(phases[phase_idx], (p, t), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1, 0].set_xlabel('Parameters (Millions)', fontweight='bold')
    axes[1, 0].set_ylabel('Test Accuracy (%)', fontweight='bold')
    axes[1, 0].set_title('Model Size vs Performance')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # 5. Training Time vs Performance
    time_clean = [(t, a) for t, a in zip(training_times, test_accs) if t is not None and a is not None]
    if time_clean:
        t_vals, a_vals = zip(*time_clean)
        axes[1, 1].scatter(t_vals, a_vals, s=200, alpha=0.6, color='purple', edgecolors='black', linewidth=2)
        
        for i, (t, a) in enumerate(time_clean):
            time_idx = [j for j, x in enumerate(training_times) if x == t][0]
            axes[1, 1].annotate(phases[time_idx], (t, a),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1, 1].set_xlabel('Training Time (hours)', fontweight='bold')
    axes[1, 1].set_ylabel('Test Accuracy (%)', fontweight='bold')
    axes[1, 1].set_title('Training Time vs Performance')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # 6. Summary Statistics Table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    
    summary_data = []
    for i, phase in enumerate(phases):
        if test_accs[i] is not None:
            summary_data.append([
                phase,
                f"{test_accs[i]:.2f}%",
                f"{perplexities[i]:.1f}" if perplexities[i] else "N/A",
                f"{overfitting_gaps[i]:.2f}%" if overfitting_gaps[i] is not None else "N/A"
            ])
    
    if summary_data:
        table = axes[1, 2].table(cellText=summary_data,
                                colLabels=['Phase', 'Test Acc', 'Perplexity', 'Overfit Gap'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    
    axes[1, 2].set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('plots/james_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: plots/james_comprehensive_comparison.png")
    plt.close()

if __name__ == "__main__":
    james_file = sys.argv[1] if len(sys.argv) > 1 else None
    create_comparison_visualizations(james_file)
