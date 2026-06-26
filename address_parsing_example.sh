#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 0-02:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=address_parsing_example
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

# Parse addresses using quantized version of gemma-4-12B-it
python3.12 address_parsing_example.py "cyankiwi/gemma-4-12B-it-AWQ-INT4"

# Deactivate environment
conda deactivate
