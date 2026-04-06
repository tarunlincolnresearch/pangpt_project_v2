#!/bin/bash
#SBATCH --job-name=temp_test
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/temp_test_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/temp_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --constraint=nvidia_rtx_a6000
#SBATCH --qos=normal

echo "========================================="
echo "Testing Different Temperatures"
echo "========================================="

for temp in 0.5 0.7 0.9 1.2 1.5; do
  echo ""
  echo "========================================="
  echo "Temperature: $temp"
  echo "========================================="
  /work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
    /work/users/tgangil/pangpt_project_v2/panGPT/panPrompt.py \
    --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
    --model_type transformer \
    --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
    --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/prompt.txt \
    --num_tokens 20 \
    --temperature $temp \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 512 \
    --device cuda
  echo ""
done

echo ""
echo "========================================="
echo "Testing Different Prompts (temp=1.0)"
echo "========================================="

for prompt_file in prompt2.txt prompt3.txt; do
  echo ""
  echo "Prompt file: $prompt_file"
  echo "---"
  /work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
    /work/users/tgangil/pangpt_project_v2/panGPT/panPrompt.py \
    --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
    --model_type transformer \
    --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
    --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/$prompt_file \
    --num_tokens 20 \
    --temperature 1.0 \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 512 \
    --device cuda
  echo ""
done
