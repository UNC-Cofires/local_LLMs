#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -t 0-01:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=local_llm_example
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --mail-type=all

# Load configuration file
source config.sh

# Load anaconda module
module purge
module load anaconda

# Activate environment
conda activate $LLM_CONDA_ENV_PATH

# Export environment variables used within python script
export HF_HOME=$HF_HOME
export HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN
export PYTHONWARNINGS="ignore"

# Load cuda module
module load $CUDA_MODULE_VERSION

# Run example python script that uses a local LLM to analyze news article text
python3.12 run_example.py

# Deactivate environment
conda deactivate
