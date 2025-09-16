#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH -o LOG_train_models_%j.out
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00

# NOTE: Run `conda activate train-env` before running this script

# Run the Python command passed as arguments to sbatch
python "$@"