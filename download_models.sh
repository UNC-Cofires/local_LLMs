#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=download_models
#SBATCH --mail-user=kieranf@email.unc.edu

# Load configuration file
source config.sh

# Load anaconda module
module purge
module load anaconda

# Activate environment
conda activate $LLM_CONDA_ENV_PATH

# Export environment variables that python will need to download models through Hugging Face
export HF_HOME=$HF_HOME
export HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Download pre-trained models
# (python script expects there to be a list of LLMs to download in a file named model_list.txt)
python3.12 download_models.py

# Deactivate environemnt
conda deactivate

