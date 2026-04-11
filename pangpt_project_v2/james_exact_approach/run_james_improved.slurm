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
# Optimized for RTX A6000 GPU (47GB memory)
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

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

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
echo ""

# ── Verify Python and PyTorch ─────────────────────────────────────────────────
echo "Python version:"
$PYTHON --version
echo ""
echo "PyTorch info:"
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch check skipped"
echo "======================================"
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
echo "Starting training with optimized parameters..."
echo "Batch size reduced to 8 to fit in GPU memory"
echo ""

$PYTHON -u panGPT.py \
    --input_file data/all_genomes.txt \
    --model_type transformer \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 1024 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --lr_scheduler_factor 0.5 \
    --lr_patience 10 \
    --weight_decay 0.0001 \
    --early_stop_patience 20 \
    --min_delta 0.01 \
    --train_size 0.8 \
    --val_size 0.1 \
    --max_vocab_size 70000 \
    --model_dropout_rate 0.2 \
    --pe_max_len 5000 \
    --pe_dropout_rate 0.1 \
    --seed 42 \
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
echo "FINAL MEMORY USAGE"
echo "======================================"
echo "  RAM : $(free -g | awk '/Mem:/{print $3"GB / "$2"GB"}')"
echo "  GPU : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1" MB / "$2" MB"}')"
echo "======================================"
echo "Training completed: $(date)"
echo "Exit code: $TRAIN_EXIT"
echo "======================================"

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Model saved to: checkpoints/james_model.pth"
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT"
    echo "Check error log: logs/james_full_${SLURM_JOB_ID}.err"
fi

exit $TRAIN_EXIT