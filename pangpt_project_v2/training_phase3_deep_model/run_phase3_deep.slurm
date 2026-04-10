#!/bin/bash
#SBATCH --job-name=phase3_deep_model
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/training_phase3_deep_model/logs/phase3_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/training_phase3_deep_model/logs/phase3_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --qos=long
#SBATCH --mem=240G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1

# ============================================================
# SLURM Phase 3 Deep Model Training
# 12 layers, 512 dim, 16 heads, ~120M parameters
# ============================================================

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
PROJECT_DIR=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2
TRAIN_SCRIPT="$PROJECT_DIR/training_phase3_deep_model/train_deep.py"

DATA_DIR="$PROJECT_DIR/data/phase1/win256"

echo "======================================"
echo "PHASE 3 - DEEP MODEL TRAINING"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""
echo "Model Architecture:"
echo "  Layers: 12 (2x deeper than Phase 2A)"
echo "  Embedding: 512 (2x wider)"
echo "  Attention heads: 16 (2x more)"
echo "  Parameters: ~120M (4x larger)"
echo ""
echo "Training Configuration:"
echo "  Batch size: 4 (with gradient accumulation)"
echo "  Effective batch: 8"
echo "  Max epochs: 150"
echo "  Expected time: 3-4 days"
echo ""
echo "Regularization (Moderate):"
echo "  Label smoothing: 0.05"
echo "  Weight decay: 0.0005"
echo "  Dropout: 0.15"
echo "  Gradient clipping: 1.0"
echo "======================================"

module purge
module load gcc
module load nvidia/cuda

nvidia-smi

cd "$PROJECT_DIR/training_phase3_deep_model"
mkdir -p checkpoints logs results

# Memory monitor
memory_monitor() {
    while true; do
        RAM_USED=$(free -g | awk '/Mem:/{print $3}')
        RAM_TOTAL=$(free -g | awk '/Mem:/{print $2}')
        GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null \
                   | awk -F', ' '{print $1" MB / "$2" MB"}')
        echo "$(date '+%Y-%m-%d %H:%M:%S')  MEMORY [Phase3]  RAM: ${RAM_USED}GB / ${RAM_TOTAL}GB  |  GPU: ${GPU_INFO}"
        sleep 300
    done
}
memory_monitor &
MONITOR_PID=$!

# Training with Deep Model
$PYTHON -u "$TRAIN_SCRIPT" \
    --train_file "$DATA_DIR/train_windows.txt" \
    --val_file "$DATA_DIR/val_windows.txt" \
    --test_file "$DATA_DIR/test_windows.txt" \
    --model_type transformer \
    --embed_dim 512 \
    --num_heads 16 \
    --num_layers 12 \
    --max_seq_length 512 \
    --batch_size 4 \
    --epochs 150 \
    --learning_rate 0.0001 \
    --lr_scheduler_factor 0.5 \
    --lr_patience 15 \
    --weight_decay 0.0005 \
    --gradient_clip 1.0 \
    --model_dropout_rate 0.15 \
    --early_stop_patience 30 \
    --min_delta 0.001 \
    --max_vocab_size 50000 \
    --pe_max_len 5000 \
    --pe_dropout_rate 0.1 \
    --model_save_path "checkpoints/model_checkpoint.pth" \
    --log_dir "logs" \
    --tokenizer_file "$PROJECT_DIR/pangenome_gpt_tokenizer.json"

TRAIN_EXIT=$?

# Stop memory monitor
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "======================================"
echo "FINAL MEMORY"
echo "  RAM : $(free -g | awk '/Mem:/{print $3"GB / "$2"GB"}')"
echo "  GPU : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1" MB / "$2" MB"}')"
echo "======================================"
echo "Finished: $(date)"
echo "Exit code: $TRAIN_EXIT"
echo "======================================"

exit $TRAIN_EXIT
