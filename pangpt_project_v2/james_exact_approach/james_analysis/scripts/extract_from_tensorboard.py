import sys
import os

try:
    from tensorboard.backend.event_processing import event_accumulator
    import json
    
    def extract_from_tensorboard(log_dir):
        """Extract metrics from TensorBoard event files"""
        
        # Find all event files
        event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
        
        if not event_files:
            print("No TensorBoard event files found")
            return None
        
        # Use the most recent event file
        event_file = os.path.join(log_dir, sorted(event_files)[-1])
        print(f"Reading: {event_file}")
        
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        metrics = {
            'training': {'final_loss': None, 'final_accuracy': None, 'epochs_completed': None},
            'validation': {'final_loss': None, 'final_accuracy': None, 'best_accuracy': None, 'best_epoch': None},
            'test': {'accuracy': None, 'loss': None, 'perplexity': None},
            'model_config': {}
        }
        
        # Get available tags
        print("Available metrics:", ea.Tags())
        
        # Extract training metrics
        if 'train_loss' in ea.Tags()['scalars']:
            train_loss = ea.Scalars('train_loss')
            metrics['training']['final_loss'] = train_loss[-1].value
            metrics['training']['epochs_completed'] = len(train_loss)
        
        if 'train_accuracy' in ea.Tags()['scalars']:
            train_acc = ea.Scalars('train_accuracy')
            metrics['training']['final_accuracy'] = train_acc[-1].value
        
        # Extract validation metrics
        if 'val_loss' in ea.Tags()['scalars']:
            val_loss = ea.Scalars('val_loss')
            metrics['validation']['final_loss'] = val_loss[-1].value
        
        if 'val_accuracy' in ea.Tags()['scalars']:
            val_acc = ea.Scalars('val_accuracy')
            val_acc_values = [x.value for x in val_acc]
            metrics['validation']['final_accuracy'] = val_acc_values[-1]
            metrics['validation']['best_accuracy'] = max(val_acc_values)
            metrics['validation']['best_epoch'] = val_acc_values.index(max(val_acc_values)) + 1
        
        # Extract test metrics
        if 'test_accuracy' in ea.Tags()['scalars']:
            test_acc = ea.Scalars('test_accuracy')
            metrics['test']['accuracy'] = test_acc[-1].value
        
        if 'test_loss' in ea.Tags()['scalars']:
            test_loss = ea.Scalars('test_loss')
            metrics['test']['loss'] = test_loss[-1].value
        
        if 'test_perplexity' in ea.Tags()['scalars']:
            test_perp = ea.Scalars('test_perplexity')
            metrics['test']['perplexity'] = test_perp[-1].value
        
        return metrics
    
    if __name__ == "__main__":
        if len(sys.argv) < 2:
            print("Usage: python extract_from_tensorboard.py <log_directory>")
            sys.exit(1)
        
        log_dir = sys.argv[1]
        metrics = extract_from_tensorboard(log_dir)
        
        if metrics:
            with open('outputs/james_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            print("\nExtracted metrics:")
            print(json.dumps(metrics, indent=2))
            print("\nSaved to: outputs/james_metrics.json")

except ImportError:
    print("Error: tensorboard package not installed")
    print("Install with: pip install tensorboard")
    sys.exit(1)
