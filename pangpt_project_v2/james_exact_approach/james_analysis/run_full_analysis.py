#!/usr/bin/env python3
"""
Master script to run complete James model analysis
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Warning: {description} failed with code {result.returncode}")
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_full_analysis.py <path_to_james_log_file>")
        print("\nThis will:")
        print("  1. Extract metrics from training log")
        print("  2. Create comparison table with all phases")
        print("  3. Generate visualizations")
        print("  4. Create comprehensive analysis report")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"\n{'#'*80}")
    print(f"# JAMES MODEL - COMPREHENSIVE ANALYSIS")
    print(f"# Log file: {log_file}")
    print(f"{'#'*80}\n")
    
    # Step 1: Extract metrics
    run_command(
        f"python scripts/extract_metrics.py {log_file}",
        "Step 1: Extracting metrics from training log"
    )
    
    # Step 2: Create comparison table
    run_command(
        "python scripts/create_comparison.py outputs/james_metrics.json",
        "Step 2: Creating comparison table"
    )
    
    # Step 3: Generate visualizations
    run_command(
        "python scripts/create_visualizations.py outputs/james_metrics.json",
        "Step 3: Generating visualizations"
    )
    
    # Step 4: Generate report
    run_command(
        "python scripts/generate_report.py outputs/james_metrics.json",
        "Step 4: Generating comprehensive report"
    )
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  • outputs/james_metrics.json")
    print("  • outputs/phase_comparison.csv")
    print("  • outputs/phase_comparison.txt")
    print("  • plots/james_comprehensive_comparison.png")
    print("  • outputs/james_analysis_report.txt")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
