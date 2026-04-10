#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --output=logs/test_cuda_%j.out
#SBATCH --time=00:05:00
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gpus=1

echo "======================================"
echo "Testing CUDA on GPU node"
echo "======================================"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

module purge
module load gcc
module load nvidia/cuda

echo "GPU Information:"
nvidia-smi
echo ""

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python

echo "PyTorch CUDA Test:"
$PYTHON -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
else:
    print('WARNING: CUDA not available!')
"

echo ""
echo "======================================"
echo "Test complete: $(date)"
echo "======================================"
