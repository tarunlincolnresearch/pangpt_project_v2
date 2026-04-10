# panGPT Training - James's Exact Approach

## Available Scripts

### run_james_original.slurm
- Your original script
- Time: 7 days (qos=long)
- Memory: 240G
- Epochs: 50

### run_james_improved.slurm ⭐ RECOMMENDED
- Enhanced with memory monitoring
- Time: 7 days (qos=long)
- Memory: 240G
- Epochs: 50
- Tracks RAM/GPU usage every 5 minutes

### run_james_quick_test.slurm 🧪 FOR TESTING
- Quick test run
- Time: 2 hours (qos=normal)
- Memory: 64G
- Epochs: 2 (reduced model size)

## Quick Start

### 1. Test first (RECOMMENDED):
```bash
sbatch run_james_quick_test.slurm