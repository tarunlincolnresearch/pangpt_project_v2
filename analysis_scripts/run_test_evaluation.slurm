#!/bin/bash
#SBATCH --job-name=test_eval
#SBATCH --output=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/test_eval_%j.out
#SBATCH --error=/work/users/tgangil/pangpt_project_v2/pangpt_project_v2/logs/test_eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --mem=16G

/work/users/tgangil/pangpt_project_v2/pangenome/bin/python \
  /work/users/tgangil/pangpt_project_v2/analysis_scripts/05_evaluate_test_set.py
