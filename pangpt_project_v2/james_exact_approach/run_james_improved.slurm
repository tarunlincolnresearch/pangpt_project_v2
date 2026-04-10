#!/bin/bash
#SBATCH --job-name=james_full_train
#SBATCH --output=logs/james_full_%j.out
#SBATCH --error=logs/james_full_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --qos=long
#SBATCH --mem=240G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1

# ============================================================
# James's Original Approach - Improved Version
# Usage: sbatch run_james_improved.slurm
# ============================================================

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
PROJECT_DIR=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/james_exact_approach

echo "======================================"
echo "JAMES'S ORIGINAL APPROACH - IMPROVED"
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

# ── Memory monitor — every 5 minutes in background ───────────────────────────
memory_monitor() {
    while true; do
        RAM_USED=$(free -g | awk '/Mem:/{print $3}')
        RAM_TOTAL=$(free -g | awk '/Mem:/{print $2}')
        GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null \
                   | awk -F', ' '{print $1" MB / "$2" MB"}')
        echo "$(date '+%Y-%m-%d %H:%M:%S')  MEMORY  RAM: ${RAM_USED}GB / ${RAM_TOTAL}GB  |  GPU: ${GPU_INFO}"
        sleep 300
    done
}
memory_monitor &
MONITOR_PID=$!
echo "Memory monitor started (PID=$MONITOR_PID)"

# ── Verify Python and PyTorch ─────────────────────────────────────────────────
echo ""
echo "Python version:"
$PYTHON --version
echo ""
echo "PyTorch info:"
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch check skipped"
echo "======================================"

# ── Training ──────────────────────────────────────────────────────────────────
echo "Starting training..."
$PYTHON -u panGPT.py \
    --input_file data/all_genomes.txt \
    --model_type transformer \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 5089 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --train_size 0.8 \
    --val_size 0.1 \
    --max_vocab_size 70000 \
    --model_save_path checkpoints/james_model.pth \
    --tokenizer_file tokenizer.json \
    --log_dir logs

TRAIN_EXIT=$?

# ── Stop memory monitor ───────────────────────────────────────────────────────
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

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