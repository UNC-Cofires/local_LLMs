#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=create_environment
#SBATCH --mail-user=kieranf@email.unc.edu

# Load configuration file
source config.sh

# Load anaconda module
module purge
module load anaconda

# Create a fresh conda environment
conda create --prefix=$LLM_CONDA_ENV_PATH python=3.12 --yes

# Activate environment
conda activate $LLM_CONDA_ENV_PATH

# Install PyTorch
#
# In your configuration file, please specify the URL to a version
# of pytorch that is compatible with the version of CUDA installed
# on Longleaf.
#
# To check the version of CUDA on Longleaf follow the steps on 
# the following page: https://help.rc.unc.edu/gpumonitor/
#
# To determine the correct download URL for PyTorch, 
# please see the following page: https://pytorch.org/

pip install torch --index-url $PYTORCH_DOWNLOAD_URL

# Install LLM dependencies
pip install transformers accelerate bitsandbytes huggingface_hub numpy pandas

# Install other packages useful for scientific computing
# (Feel free to add to this list)
pip install scipy pyarrow matplotlib requests

# Optional: Register kernel so you can access the environment from a Jupyter notebook session
pip install ipykernel
python3.12 -m ipykernel install --user --name="$LLM_CONDA_ENV_NAME"

# Deactivate environemnt
conda deactivate

