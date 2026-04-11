#!/bin/bash

echo "================================================================================================"
echo "TRAINING MODEL WITH REDUCED VOCABULARY (Threshold ≥10)"
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
echo "Model saved to: checkpoints/reduced_vocab10_model.pth"
echo "================================================================================================"
