#!/bin/bash
#SBATCH --job-name=sampling_test
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/sampling_test_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/sampling_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --constraint=nvidia_rtx_a6000
#SBATCH --qos=normal

echo "========================================="
echo "Testing Different Sampling Strategies"
echo "========================================="

# Test 1: Higher temperature with top-k/top-p
echo ""
echo "Test 1: temp=1.5, top_k=50, top_p=0.9"
echo "---"
/work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
  /work/users/tgangil/pangpt_project_v2/panGPT/panPrompt_v2.py \
  --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
  --model_type transformer \
  --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
  --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/prompt.txt \
  --num_tokens 30 \
  --temperature 1.5 \
  --top_k 50 \
  --top_p 0.9 \
  --embed_dim 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_seq_length 512 \
  --device cuda

# Test 2: Lower top_k for more diversity
echo ""
echo "Test 2: temp=1.2, top_k=30, top_p=0.9"
echo "---"
/work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
  /work/users/tgangil/pangpt_project_v2/panGPT/panPrompt_v2.py \
  --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
  --model_type transformer \
  --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
  --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/prompt.txt \
  --num_tokens 30 \
  --temperature 1.2 \
  --top_k 30 \
  --top_p 0.9 \
  --embed_dim 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_seq_length 512 \
  --device cuda

# Test 3: Only top-p (nucleus sampling)
echo ""
echo "Test 3: temp=1.3, top_k=0 (disabled), top_p=0.85"
echo "---"
/work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
  /work/users/tgangil/pangpt_project_v2/panGPT/panPrompt_v2.py \
  --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
  --model_type transformer \
  --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
  --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/prompt.txt \
  --num_tokens 30 \
  --temperature 1.3 \
  --top_k 0 \
  --top_p 0.85 \
  --embed_dim 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_seq_length 512 \
  --device cuda

# Test 4: Aggressive diversity
echo ""
echo "Test 4: temp=1.8, top_k=100, top_p=0.95"
echo "---"
/work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
  /work/users/tgangil/pangpt_project_v2/panGPT/panPrompt_v2.py \
  --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
  --model_type transformer \
  --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
  --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/prompt.txt \
  --num_tokens 30 \
  --temperature 1.8 \
  --top_k 100 \
  --top_p 0.95 \
  --embed_dim 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_seq_length 512 \
  --device cuda

echo ""
echo "========================================="
echo "Tests Complete"
echo "========================================="
