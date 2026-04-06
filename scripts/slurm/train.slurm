#!/bin/bash
#SBATCH --job-name=pangpt_train
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/train_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/train_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --qos=long
#SBATCH --mem=240G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1

# ============================================================
# SLURM Training Script — PanGPT Pipeline
# Usage:  sbatch scripts/slurm/train.slurm
# ============================================================

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
PROJECT_DIR=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2

echo "======================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "Python      : $PYTHON"
echo "Project dir : $PROJECT_DIR"
echo "======================================"

module purge
module load gcc
module load nvidia/cuda

nvidia-smi

cd "$PROJECT_DIR"

$PYTHON -u scripts/04_train.py

echo ""
echo "======================================"
echo "Finished at : $(date)"
echo "======================================"
