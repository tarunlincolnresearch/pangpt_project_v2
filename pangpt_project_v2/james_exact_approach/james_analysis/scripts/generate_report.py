import json
import sys
from datetime import datetime

def generate_james_analysis_report(james_metrics_file):
    """
    Generate comprehensive analysis report for James model
    """
    
    # Load metrics
    try:
        with open(james_metrics_file, 'r') as f:
            james_metrics = json.load(f)
    except:
        print("Error: Could not load James metrics file")
        return
    
    # Extract values
    cfg = james_metrics['model_config']
    train = james_metrics['training']
    val = james_metrics['validation']
    test = james_metrics['test']
    
    # Calculate derived metrics
    val_acc = val.get('final_accuracy', 0) or 0
    test_acc = test.get('accuracy', 0) or 0
    gap = val_acc - test_acc
    
    report = f"""
====================================================================================================
JAMES MODEL ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
====================================================================================================

CONFIGURATION
----------------------------------------------------------------------------------------------------
Architecture:
  • Layers: {cfg.get('num_layers', 'N/A')}
  • Embedding Dimension: {cfg.get('embed_dim', 'N/A')}
  • Attention Heads: {cfg.get('num_heads', 'N/A')}
  • Total Parameters: {cfg.get('total_params', 'N/A')}

Regularization:
  • Dropout: {cfg.get('dropout', 'N/A')}
  • Label Smoothing: {cfg.get('label_smoothing', 'N/A')}
  • Weight Decay: {cfg.get('weight_decay', 'N/A')}

Training:
  • Epochs: {train.get('epochs_completed', 'N/A')}
  • Training Time: {train.get('training_time', 'N/A')}
  • Final Training Loss: {train.get('final_loss', 'N/A')}
  • Final Training Accuracy: {train.get('final_accuracy', 'N/A')}%

RESULTS
----------------------------------------------------------------------------------------------------
Validation Performance:
  • Final Validation Accuracy: {val.get('final_accuracy', 'N/A')}%
  • Best Validation Accuracy: {val.get('best_accuracy', 'N/A')}%
  • Best Epoch: {val.get('best_epoch', 'N/A')}
  • Final Validation Loss: {val.get('final_loss', 'N/A')}

Test Performance:
  • Test Accuracy: {test.get('accuracy', 'N/A')}%
  • Test Loss: {test.get('loss', 'N/A')}
  • Test Perplexity: {test.get('perplexity', 'N/A')}

ANALYSIS
----------------------------------------------------------------------------------------------------
Overfitting Analysis:
  • Validation-Test Gap: {gap:.2f}%
"""
    
    # Overfitting assessment
    if gap > 10:
        report += f"  ❌ SEVERE OVERFITTING: Model memorizing training data\n"
    elif gap > 5:
        report += f"  ⚠️  MODERATE OVERFITTING: Some generalization issues\n"
    elif gap > 1:
        report += f"  ⚠️  SLIGHT OVERFITTING: Minor generalization gap\n"
    else:
        report += f"  ✅ NO OVERFITTING: Good generalization\n"
    
    # Performance assessment
    report += f"\nPerformance Assessment:\n"
    if test_acc < 10:
        report += f"  ❌ CRITICAL FAILURE: Test accuracy ({test_acc:.2f}%) near random\n"
    elif 48 <= test_acc <= 52:
        report += f"  ⚠️  50% PLATEAU CONFIRMED: Test accuracy = {test_acc:.2f}%\n"
    elif test_acc > 52:
        report += f"  ✅ BREAKTHROUGH: Test accuracy = {test_acc:.2f}%\n"
        report += f"     EXCEEDED the 50% plateau by {test_acc - 50.29:.2f}%\n"
    
    # Perplexity assessment
    perp = test.get('perplexity', 0) or 0
    report += f"\nPerplexity Analysis:\n"
    if perp < 50:
        report += f"  ⚠️  SUSPICIOUSLY LOW: Perplexity = {perp:.2f}\n"
    elif 50 <= perp <= 200:
        report += f"  ✅ GOOD RANGE: Perplexity = {perp:.2f}\n"
    else:
        report += f"  ⚠️  HIGH PERPLEXITY: Perplexity = {perp:.2f}\n"
    
    report += f"""

COMPARISON TO PREVIOUS PHASES
----------------------------------------------------------------------------------------------------
Phase 1:  Val=65.5%  Test=0.51%   Perp=2.49    [Severe Overfitting]
Phase 2:  Val=50.28% Test=50.23%  Perp=258.32  [Over-regularized]
Phase 2A: Val=50.27% Test=50.27%  Perp=142.32  [Balanced]
Phase 3:  Val=50.29% Test=50.29%  Perp=141.30  [Balanced - 3x larger]
James:    Val={val_acc:.2f}% Test={test_acc:.2f}%  Perp={perp:.2f}

====================================================================================================
END OF REPORT
====================================================================================================
"""
    
    # Save report
    with open('outputs/james_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("\nReport saved to: outputs/james_analysis_report.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <james_metrics.json>")
        sys.exit(1)
    
    generate_james_analysis_report(sys.argv[1])
