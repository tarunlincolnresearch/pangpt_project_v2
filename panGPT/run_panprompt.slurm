#!/bin/bash
#SBATCH --job-name=panprompt
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/panprompt_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/panprompt_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --constraint=nvidia_rtx_a6000
#SBATCH --qos=normal

/work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
  /work/users/tgangil/pangpt_project_v2/panGPT/panPrompt.py \
  --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
  --model_type transformer \
  --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
  --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/prompt.txt \
  --num_tokens 20 \
  --temperature 1.0 \
  --embed_dim 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_seq_length 512 \
  --device cuda
