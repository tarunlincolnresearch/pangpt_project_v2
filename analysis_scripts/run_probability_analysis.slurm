#!/bin/bash
#SBATCH --job-name=prob_analysis
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/prob_analysis_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/prob_analysis_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --qos=normal

cd /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/results

/work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
  /work/users/tgangil/pangpt_project_v2/analysis_scripts/03_probability_analysis.py \
  --model_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/checkpoints/win128/model_checkpoint.pth \
  --tokenizer_path /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/pangenome_gpt_tokenizer.json \
  --prompt_file /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/prompt.txt \
  --mode both \
  --target_gene "nanS" \
  --top_k 30 \
  --embed_dim 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_seq_length 512 \
  --device cuda
