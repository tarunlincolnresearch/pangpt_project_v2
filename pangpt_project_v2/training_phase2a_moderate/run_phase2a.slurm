#!/bin/bash
#SBATCH --job-name=phase2a_moderate
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/training_phase2a_moderate/logs/phase2a_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/training_phase2a_moderate/logs/phase2a_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --qos=long
#SBATCH --mem=240G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
PROJECT_DIR=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2
TRAIN_SCRIPT="$PROJECT_DIR/training_phase2a_moderate/train_moderate.py"

DATA_DIR="$PROJECT_DIR/data/phase1/win256"

echo "======================================"
echo "PHASE 2A - MODERATE REGULARIZATION"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""
echo "Moderate Settings:"
echo "  - Label smoothing: 0.05"
echo "  - Weight decay: 0.0005"
echo "  - Dropout: 0.15"
echo "  - Gradient clipping: 1.0"
echo "======================================"

module purge
module load gcc
module load nvidia/cuda

nvidia-smi

cd "$PROJECT_DIR/training_phase2a_moderate"
mkdir -p checkpoints logs results

$PYTHON -u "$TRAIN_SCRIPT" \
    --train_file "$DATA_DIR/train_windows.txt" \
    --val_file "$DATA_DIR/val_windows.txt" \
    --test_file "$DATA_DIR/test_windows.txt" \
    --model_type transformer \
    --embed_dim 256 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 512 \
    --batch_size 8 \
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

echo ""
echo "Finished: $(date)"
exit $?
