import pandas as pd
import json
import sys

def create_comparison_table(james_metrics_file=None):
    """
    Create comparison table for all phases including James
    """
    
    # Load James metrics if provided
    james_data = {
        'layers': None,
        'embed_dim': None,
        'heads': None,
        'params': None,
        'dropout': None,
        'label_smoothing': None,
        'weight_decay': None,
        'val_acc': None,
        'test_acc': None,
        'test_perp': None,
        'train_time': None
    }
    
    if james_metrics_file:
        try:
            with open(james_metrics_file, 'r') as f:
                james_metrics = json.load(f)
            
            james_data['layers'] = james_metrics['model_config'].get('num_layers')
            james_data['embed_dim'] = james_metrics['model_config'].get('embed_dim')
            james_data['heads'] = james_metrics['model_config'].get('num_heads')
            james_data['params'] = james_metrics['model_config'].get('total_params')
            james_data['dropout'] = james_metrics['model_config'].get('dropout')
            james_data['label_smoothing'] = james_metrics['model_config'].get('label_smoothing')
            james_data['weight_decay'] = james_metrics['model_config'].get('weight_decay')
            james_data['val_acc'] = james_metrics['validation'].get('final_accuracy')
            james_data['test_acc'] = james_metrics['test'].get('accuracy')
            james_data['test_perp'] = james_metrics['test'].get('perplexity')
            james_data['train_time'] = james_metrics['training'].get('training_time')
        except Exception as e:
            print(f"Warning: Could not load James metrics: {e}")
    
    comparison_data = {
        'Phase': ['Phase 1', 'Phase 2', 'Phase 2A', 'Phase 3', 'James'],
        
        # Architecture
        'Layers': [6, 6, 6, 12, james_data['layers']],
        'Embed_Dim': [512, 256, 256, 512, james_data['embed_dim']],
        'Heads': [8, 8, 8, 16, james_data['heads']],
        'Params': ['~50M', '~30M', '~30M', '~89M', james_data['params']],
        
        # Regularization
        'Dropout': [0.1, 0.2, 0.15, 0.15, james_data['dropout']],
        'Label_Smoothing': [0.0, 0.1, 0.05, 0.05, james_data['label_smoothing']],
        'Weight_Decay': [0.0, 0.001, 0.0005, 0.0005, james_data['weight_decay']],
        
        # Results
        'Val_Accuracy': [65.5, 50.28, 50.27, 50.29, james_data['val_acc']],
        'Test_Accuracy': [0.51, 50.23, 50.27, 50.29, james_data['test_acc']],
        'Test_Perplexity': [2.49, 258.32, 142.32, 141.30, james_data['test_perp']],
        
        # Training
        'Training_Time': ['~6h', '~6h', '~6h', '~27h', james_data['train_time']],
    }
    
    # Calculate overfitting gap
    overfitting_gaps = []
    for val, test in zip(comparison_data['Val_Accuracy'], comparison_data['Test_Accuracy']):
        if val is not None and test is not None:
            overfitting_gaps.append(round(val - test, 2))
        else:
            overfitting_gaps.append(None)
    
    comparison_data['Overfitting_Gap'] = overfitting_gaps
    
    # Status
    statuses = ['Severe Overfit', 'Over-regularized', 'Balanced', 'Balanced']
    
    # Determine James status
    if james_data['test_acc'] is not None:
        if overfitting_gaps[-1] and overfitting_gaps[-1] > 10:
            james_status = 'Overfitting'
        elif james_data['test_acc'] < 10:
            james_status = 'Failed'
        elif 48 <= james_data['test_acc'] <= 52:
            james_status = '50% Plateau'
        elif james_data['test_acc'] > 52:
            james_status = 'Breakthrough!'
        else:
            james_status = 'Underperforming'
    else:
        james_status = 'TBD'
    
    statuses.append(james_status)
    comparison_data['Status'] = statuses
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    df.to_csv('outputs/phase_comparison.csv', index=False)
    
    # Save formatted text version
    with open('outputs/phase_comparison.txt', 'w') as f:
        f.write("="*120 + "\n")
        f.write("PHASE COMPARISON TABLE\n")
        f.write("="*120 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*120 + "\n")
    
    print(df.to_string(index=False))
    print("\nComparison saved to:")
    print("  - outputs/phase_comparison.csv")
    print("  - outputs/phase_comparison.txt")
    
    return df

if __name__ == "__main__":
    james_file = sys.argv[1] if len(sys.argv) > 1 else None
    create_comparison_table(james_file)
