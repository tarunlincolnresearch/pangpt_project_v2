#!/usr/bin/env python3
"""
Plot all training metrics from TensorBoard logs
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import argparse

def extract_scalar_data(log_dir, tag):
    """Extract scalar data from TensorBoard logs"""
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None
    
    # Use the directory (not individual files) for EventAccumulator
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Check if tag exists
    if tag not in ea.Tags()['scalars']:
        print(f"Tag '{tag}' not found. Available tags: {ea.Tags()['scalars']}")
        return None, None
    
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return steps, values

def plot_metric(log_dir, tag, title, ylabel, output_file, color='blue'):
    """Plot a single metric"""
    steps, values = extract_scalar_data(log_dir, tag)
    
    if steps is None or values is None:
        print(f"Skipping {title} - no data found")
        return False
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, values, marker='o', linestyle='-', color=color, linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")
    return True

def plot_comparison(log_dir, train_tag, val_tag, title, ylabel, output_file):
    """Plot training vs validation comparison"""
    train_steps, train_values = extract_scalar_data(log_dir, train_tag)
    val_steps, val_values = extract_scalar_data(log_dir, val_tag)
    
    if train_steps is None or val_steps is None:
        print(f"Skipping {title} - missing data")
        return False
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_values, marker='o', linestyle='-', 
             color='blue', linewidth=2, markersize=4, label='Training', alpha=0.8)
    plt.plot(val_steps, val_values, marker='s', linestyle='-', 
             color='red', linewidth=2, markersize=4, label='Validation', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory for plots')
    args = parser.parse_args()
    
    log_dir = args.log_dir
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("JAMES MODEL - TRAINING METRICS VISUALIZATION")
    print("="*80)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Plot individual metrics
    metrics = [
        # Training metrics
        ('Loss/train', 'Training Loss Over Epochs', 'Loss', 'training_loss.png', 'blue'),
        ('Perplexity/train', 'Training Perplexity Over Epochs', 'Perplexity', 'training_perplexity.png', 'blue'),
        
        # Validation metrics
        ('Loss/val', 'Validation Loss Over Epochs', 'Loss', 'validation_loss.png', 'red'),
        ('Perplexity/val', 'Validation Perplexity Over Epochs', 'Perplexity', 'validation_perplexity.png', 'red'),
        ('Accuracy/val', 'Validation Accuracy Over Epochs', 'Accuracy', 'validation_accuracy.png', 'green'),
        ('Precision/val', 'Validation Precision Over Epochs', 'Precision', 'validation_precision.png', 'purple'),
        ('Recall/val', 'Validation Recall Over Epochs', 'Recall', 'validation_recall.png', 'orange'),
        ('F1/val', 'Validation F1 Score Over Epochs', 'F1 Score', 'validation_f1.png', 'brown'),
        ('Kappa/val', 'Validation Cohen\'s Kappa Over Epochs', 'Kappa', 'validation_kappa.png', 'pink'),
        
        # Learning rate
        ('Learning Rate', 'Learning Rate Schedule', 'Learning Rate', 'learning_rate.png', 'black'),
    ]
    
    print("Plotting individual metrics...")
    for tag, title, ylabel, filename, color in metrics:
        output_file = os.path.join(output_dir, filename)
        plot_metric(log_dir, tag, title, ylabel, output_file, color)
    
    print("\nPlotting comparison metrics...")
    # Comparison plots
    comparisons = [
        ('Loss/train', 'Loss/val', 'Training vs Validation Loss', 'Loss', 'loss_comparison.png'),
        ('Perplexity/train', 'Perplexity/val', 'Training vs Validation Perplexity', 'Perplexity', 'perplexity_comparison.png'),
    ]
    
    for train_tag, val_tag, title, ylabel, filename in comparisons:
        output_file = os.path.join(output_dir, filename)
        plot_comparison(log_dir, train_tag, val_tag, title, ylabel, output_file)
    
    print("\n" + "="*80)
    print("✓ All plots generated successfully!")
    print(f"✓ Plots saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
