#!/bin/bash
#SBATCH --job-name=pangpt_phase1_%a
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/phase1_%a_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/phase1_%a_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --qos=long
#SBATCH --mem=240G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --array=0-4

# ============================================================
# SLURM Phase 1 Array Job — one GPU job per window size
# Usage:  sbatch scripts/slurm/train_phase1.slurm
# ============================================================

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
PROJECT_DIR=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2
PANGPT=/home/users/tgangil/panGPT/panGPT.py

LABELS=(win128 win256 win512 win1024 win2048)
WINDOWS=(128 256 512 1024 2048)

LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
WINDOW=${WINDOWS[$SLURM_ARRAY_TASK_ID]}

DATA="$PROJECT_DIR/data/phase1/$LABEL"

echo "======================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Array ID   : $SLURM_ARRAY_TASK_ID"
echo "Experiment : $LABEL  (window=$WINDOW)"
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

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/checkpoints/$LABEL"
mkdir -p "$PROJECT_DIR/logs/$LABEL"

# ── Memory monitor — every 5 minutes in background ───────────────────────────
memory_monitor() {
    while true; do
        RAM_USED=$(free -g | awk '/Mem:/{print $3}')
        RAM_TOTAL=$(free -g | awk '/Mem:/{print $2}')
        GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null \
                   | awk -F', ' '{print $1" MB / "$2" MB"}')
        echo "$(date '+%Y-%m-%d %H:%M:%S')  MEMORY [$LABEL]  RAM: ${RAM_USED}GB / ${RAM_TOTAL}GB  |  GPU: ${GPU_INFO}"
        sleep 300
    done
}
memory_monitor &
MONITOR_PID=$!
echo "Memory monitor started (PID=$MONITOR_PID)"

# ── Training ──────────────────────────────────────────────────────────────────
$PYTHON -u "$PANGPT" \
    --input_file          "$DATA/train_windows.txt" \
    --model_type          transformer \
    --embed_dim           512 \
    --num_heads           8 \
    --num_layers          6 \
    --max_seq_length      "$WINDOW" \
    --batch_size          4 \
    --epochs              150 \
    --learning_rate       0.0001 \
    --lr_scheduler_factor 0.5 \
    --lr_patience         15 \
    --weight_decay        0.0001 \
    --early_stop_patience 30 \
    --min_delta           0.001 \
    --max_vocab_size      50000 \
    --model_dropout_rate  0.1 \
    --train_size          0.999 \
    --val_size            0.001 \
    --seed                42 \
    --pe_max_len          5000 \
    --pe_dropout_rate     0.1 \
    --model_save_path     "$PROJECT_DIR/checkpoints/$LABEL/model_checkpoint.pth" \
    --log_dir             "$PROJECT_DIR/logs/$LABEL" \
    --tokenizer_file      "$PROJECT_DIR/pangenome_gpt_tokenizer.json"

TRAIN_EXIT=$?

# ── Stop memory monitor ───────────────────────────────────────────────────────
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

# ── Final snapshot ────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "FINAL MEMORY [$LABEL]"
echo "  RAM : $(free -g | awk '/Mem:/{print $3"GB / "$2"GB"}')"
echo "  GPU : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1" MB / "$2" MB"}')"
echo "======================================"
echo "Finished  : $(date)"
echo "Exit code : $TRAIN_EXIT"
echo "======================================"

exit $TRAIN_EXIT
