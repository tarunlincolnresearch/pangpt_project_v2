#!/bin/bash
#SBATCH --job-name=james_quick_test
#SBATCH --output=logs/james_test_%j.out
#SBATCH --error=logs/james_test_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1

# ============================================================
# James's Approach - Quick Test (2 hours, 2 epochs)
# Usage: sbatch run_james_quick_test.slurm
# ============================================================

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
PROJECT_DIR=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/james_exact_approach

echo "======================================"
echo "JAMES'S APPROACH - QUICK TEST"
echo "======================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Project    : $PROJECT_DIR"
echo "Python     : $PYTHON"
echo "Start      : $(date)"
echo "======================================"

module purge
module load gcc
module load nvidia/cuda

nvidia-smi

cd "$PROJECT_DIR"

# ── Training with reduced parameters ──────────────────────────────────────────
echo "Starting QUICK TEST with reduced parameters..."
$PYTHON -u panGPT.py \
    --input_file data/all_genomes.txt \
    --model_type transformer \
    --embed_dim 256 \
    --num_heads 4 \
    --num_layers 3 \
    --max_seq_length 1024 \
    --batch_size 16 \
    --epochs 2 \
    --learning_rate 0.0001 \
    --train_size 0.8 \
    --val_size 0.1 \
    --max_vocab_size 50000 \
    --model_save_path checkpoints/james_test_model.pth \
    --tokenizer_file tokenizer_test.json \
    --log_dir logs

TRAIN_EXIT=$?

# ── Final snapshot ────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "FINAL MEMORY"
echo "  RAM : $(free -g | awk '/Mem:/{print $3"GB / "$2"GB"}')"
echo "  GPU : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1" MB / "$2" MB"}')"
echo "======================================"
echo "Finished  : $(date)"
echo "Exit code : $TRAIN_EXIT"
echo "======================================"

exit $TRAIN_EXIT
