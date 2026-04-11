#!/bin/bash
#SBATCH --job-name=vocab10_train
#SBATCH --output=logs/training_vocab10_%j.out
#SBATCH --error=logs/training_vocab10_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TGangil@lincoln.ac.uk
#SBATCH --qos=long

echo "================================================================================================"
echo "TRAINING MODEL WITH REDUCED VOCABULARY (Threshold ≥10)"
echo "================================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================================================"

# Load environment
module purge
module load anaconda3
source activate pangenome

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  CUDA: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"

# Navigate to directory
cd /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/james_improved_vocab_reduction

# Create necessary directories
mkdir -p checkpoints logs

# Run training
echo ""
echo "Starting training..."
echo "================================================================================================"

python scripts/train_reduced_vocab.py \
    --input_file data/all_genomes_vocab10.txt \
    --model_type transformer \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 4096 \
    --batch_size 32 \
    --model_dropout_rate 0.2 \
    --learning_rate 0.0001 \
    --weight_decay 1e-4 \
    --early_stop_patience 20 \
    --epochs 50 \
    --max_vocab_size 70000 \
    --model_save_path checkpoints/reduced_vocab10_model.pth \
    --tokenizer_file tokenizer_vocab10.json \
    --train_size 0.8 \
    --val_size 0.1 \
    --seed 42 \
    --log_dir logs

echo ""
echo "================================================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "Model saved to: checkpoints/reduced_vocab10_model.pth"
echo "================================================================================================"

# Print final memory usage
echo ""
echo "Final memory usage:"
free -h
nvidia-smi

