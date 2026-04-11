import re
import json
import sys
import math

def extract_james_metrics(log_file_path):
    """
    Extract key metrics from training logs
    """
    metrics = {
        'training': {
            'final_loss': None,
            'final_accuracy': None,
            'epochs_completed': None,
            'training_time': None
        },
        'validation': {
            'final_loss': None,
            'final_accuracy': None,
            'best_accuracy': None,
            'best_epoch': None
        },
        'test': {
            'accuracy': None,
            'loss': None,
            'perplexity': None
        },
        'model_config': {
            'num_layers': None,
            'embed_dim': None,
            'num_heads': None,
            'total_params': None,
            'dropout': None,
            'label_smoothing': None,
            'weight_decay': None
        }
    }
    
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        # Extract epoch-by-epoch data
        epoch_pattern = r'Epoch (\d+) - (?:Training|Validation).*?Loss: ([\d.]+).*?Perplexity: ([\d.]+).*?(?:Accuracy: ([\d.]+))?'
        epochs = re.findall(epoch_pattern, log_content, re.DOTALL)
        
        # Training metrics
        train_loss_pattern = r'Epoch \d+ - Training Loss: ([\d.]+)'
        train_losses = re.findall(train_loss_pattern, log_content)
        if train_losses:
            metrics['training']['final_loss'] = float(train_losses[-1])
        
        # Validation metrics
        val_loss_pattern = r'Epoch (\d+) - Validation Loss: ([\d.]+), Perplexity: ([\d.]+), Accuracy: ([\d.]+)'
        val_matches = re.findall(val_loss_pattern, log_content)
        
        if val_matches:
            val_accs = [float(m[3]) * 100 for m in val_matches]  # Convert to percentage
            metrics['validation']['final_accuracy'] = val_accs[-1]
            metrics['validation']['best_accuracy'] = max(val_accs)
            metrics['validation']['best_epoch'] = val_accs.index(max(val_accs)) + 1
            metrics['validation']['final_loss'] = float(val_matches[-1][1])
        
        # Test metrics
        test_pattern = r'Test Loss: ([\d.]+), Perplexity: ([\d.]+), Accuracy: ([\d.]+)'
        test_match = re.search(test_pattern, log_content)
        
        if test_match:
            metrics['test']['loss'] = float(test_match.group(1))
            metrics['test']['perplexity'] = float(test_match.group(2))
            metrics['test']['accuracy'] = float(test_match.group(3)) * 100  # Convert to percentage
        
        # Epochs
        epoch_nums = re.findall(r'Epoch (\d+) -', log_content)
        if epoch_nums:
            metrics['training']['epochs_completed'] = max([int(x) for x in epoch_nums])
        
        # Model configuration
        config_patterns = {
            'num_layers': r'(?:num_layers|n_layers|layers)[:\s=]+(\d+)',
            'embed_dim': r'(?:embed_dim|embedding_dim|d_model)[:\s=]+(\d+)',
            'num_heads': r'(?:num_heads|n_heads|heads)[:\s=]+(\d+)',
            'dropout': r'dropout[:\s=]+([\d.]+)',
            'label_smoothing': r'label_smoothing[:\s=]+([\d.]+)',
            'weight_decay': r'weight_decay[:\s=]+([\d.]+)',
        }
        
        for key, pattern in config_patterns.items():
            match = re.search(pattern, log_content, re.IGNORECASE)
            if match:
                value = match.group(1)
                metrics['model_config'][key] = float(value) if '.' in value else int(value)
        
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <log_file_path>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    metrics = extract_james_metrics(log_file)
    
    # Save to JSON
    with open('outputs/james_metrics.json', 'w') as f:
        json.dump(metrics, indent=2, fp=f)
    
    print("Extracted Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nMetrics saved to: outputs/james_metrics.json")