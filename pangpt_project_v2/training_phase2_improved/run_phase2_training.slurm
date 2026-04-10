#!/bin/bash
#SBATCH --job-name=pangpt_phase2_win256
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/training_phase2_improved/logs/phase2_win256_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/training_phase2_improved/logs/phase2_win256_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --qos=long
#SBATCH --mem=240G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1

# ============================================================
# SLURM Phase 2 Improved Training — WIN256 with enhancements
# Uses pre-split genome-level train/val/test files
# ============================================================

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
PROJECT_DIR=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2
TRAIN_SCRIPT="$PROJECT_DIR/training_phase2_improved/train_improved.py"

LABEL=win256
DATA_DIR="$PROJECT_DIR/data/phase1/$LABEL"

echo "======================================"
echo "PHASE 2 IMPROVED TRAINING"
echo "======================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Window     : $LABEL"
echo "Node       : $SLURMD_NODENAME"
echo "Start      : $(date)"
echo ""
echo "Data Files (Pre-split at genome level):"
echo "  Train: $DATA_DIR/train_windows.txt"
echo "  Val:   $DATA_DIR/val_windows.txt"
echo "  Test:  $DATA_DIR/test_windows.txt"
echo ""
echo "Phase 2 Improvements:"
echo "  ✓ Label smoothing: 0.1"
echo "  ✓ Weight decay: 1e-3 (up from 1e-4)"
echo "  ✓ Gradient clipping: 1.0"
echo "  ✓ Dropout: 0.2"
echo "  ✓ Using pre-split files (no internal split)"
echo "======================================"

module purge
module load gcc
module load nvidia/cuda

nvidia-smi

cd "$PROJECT_DIR/training_phase2_improved"

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p checkpoints logs results

# ── Memory monitor ────────────────────────────────────────────────────────────
memory_monitor() {
    while true; do
        RAM_USED=$(free -g | awk '/Mem:/{print $3}')
        RAM_TOTAL=$(free -g | awk '/Mem:/{print $2}')
        GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null \
                   | awk -F', ' '{print $1" MB / "$2" MB"}')
        echo "$(date '+%Y-%m-%d %H:%M:%S')  MEMORY [Phase2]  RAM: ${RAM_USED}GB / ${RAM_TOTAL}GB  |  GPU: ${GPU_INFO}"
        sleep 300
    done
}
memory_monitor &
MONITOR_PID=$!

# ── Training with Phase 2 Improvements ───────────────────────────────────────
$PYTHON -u "$TRAIN_SCRIPT" \
    --train_file          "$DATA_DIR/train_windows.txt" \
    --val_file            "$DATA_DIR/val_windows.txt" \
    --test_file           "$DATA_DIR/test_windows.txt" \
    --model_type          transformer \
    --embed_dim           256 \
    --num_heads           8 \
    --num_layers          6 \
    --max_seq_length      512 \
    --batch_size          8 \
    --epochs              150 \
    --learning_rate       0.0001 \
    --lr_scheduler_factor 0.5 \
    --lr_patience         15 \
    --weight_decay        0.001 \
    --gradient_clip       1.0 \
    --early_stop_patience 30 \
    --min_delta           0.001 \
    --max_vocab_size      50000 \
    --model_dropout_rate  0.2 \
    --pe_max_len          5000 \
    --pe_dropout_rate     0.1 \
    --model_save_path     "checkpoints/model_checkpoint.pth" \
    --log_dir             "logs" \
    --tokenizer_file      "$PROJECT_DIR/pangenome_gpt_tokenizer.json"

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
