#!/bin/bash
#SBATCH --job-name=pangpt_hparam
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/hparam_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/hparam_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1

# ============================================================
# SLURM Hyperparameter Search — PanGPT Pipeline
# Usage:  sbatch scripts/slurm/hyperparam.slurm
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

$PYTHON -u scripts/03_hyperparam_search.py \
    --input_file "data/phase1/win512/train_windows.txt"

echo "Finished at : $(date)"
