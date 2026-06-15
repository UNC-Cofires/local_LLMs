# Running Local LLMs on UNC's Longleaf HPC Cluster

This repository provides scripts for downloading and running large language
models (LLMs) locally on UNC's [Longleaf HPC cluster](https://help.rc.unc.edu/longleaf-cluster/).
It includes a working example that uses a local LLM to classify news articles
as describing a flood or hurricane disaster event.

---

## Overview

Running LLMs locally on Longleaf is useful when:
- You are working with sensitive data that should not be sent to an external
  API
- You want to reduce research expenses associated with commercial API usage,
  particularly when processing large volumes of text
- You want full control over the model and inference settings

This repository uses the [Hugging Face Transformers](https://huggingface.co/docs/transformers)
library to load and run models, and [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
for 4-bit quantization, which reduces the GPU memory required to run large
models.

---

## Prerequisites

### Longleaf Access
You will need an active Longleaf account. If you do not have one, visit the
[ITS Research Computing Technical Documentation](https://help.rc.unc.edu/) page for
information on how to request access.

### Hugging Face Account and Access Token
Models are downloaded from the [Hugging Face Hub](https://huggingface.co),
which requires a free account and an access token.

1. Create a Hugging Face account at https://huggingface.co
2. Generate a read-only access token at https://huggingface.co/settings/tokens
3. Save your token to a file on Longleaf:
   ```bash
   echo "your_token_here" > ~/.hf_token
   chmod 600 ~/.hf_token
   ```

### Model License Agreements
Some models require you to accept a license agreement on Hugging Face before
downloading. Make sure you are logged in to Hugging Face and have accepted
the license for each model you intend to use:
- **Llama 3.1:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **Gemma 4:** https://huggingface.co/google/gemma-4-12B-it

---

## Repository Contents

| File | Description |
|---|---|
| `config.sh.example` | Template configuration file. Copy and edit this to create your own `config.sh`. |
| `create_environment.sh` | SLURM job script that creates a conda environment and installs all dependencies. |
| `download_models.sh` | SLURM job script that downloads models listed in `model_list.txt`. |
| `download_models.py` | Python script called by `download_models.sh` to perform the downloads. |
| `model_list.txt` | Plain text list of Hugging Face model IDs to download, one per line. |
| `run_example.sh` | SLURM job script that runs the example classification task. |
| `run_example.py` | Python script that classifies a news article using a local LLM. |
| `example_text.md` | Example news article used as input for the classification task. |
| `example-slurm.out` | Example output from a successful run of `run_example.sh`. |

---

## Setup

Setup only needs to be performed once. Steps 1 and 2 can be run on a general (CPU-only) compute node -- a GPU node is not required.

### Step 1: Create `config.sh`

Copy the example configuration file and edit it with your own settings:

```bash
cp config.sh.example config.sh
```

Open `config.sh` and fill in the following variables:

| Variable | Description |
|---|---|
| `LLM_CONDA_ENV_PATH` | Full path where the conda environment will be created |
| `LLM_CONDA_ENV_NAME` | Name of the conda environment |
| `PYTORCH_DOWNLOAD_URL` | PyTorch download URL matching the CUDA version on Longleaf's GPU nodes (see note below) |
| `HF_HOME` | Directory for Hugging Face cache files -- use a `/proj` directory to avoid exceeding your home directory quota |
| `HUGGING_FACE_HUB_TOKEN` | Your Hugging Face read-only access token |

> **Note on PyTorch version:** The correct `PYTORCH_DOWNLOAD_URL` depends on
> the version of CUDA installed on Longleaf's GPU nodes. To check the CUDA
> version, run `nvidia-smi` during a brief [interactive GPU session](https://help.rc.unc.edu/gpumonitor/). Then visit
> https://pytorch.org to find the matching download URL. If your CUDA version
> falls between two available PyTorch options, choose the closest version that
> is less than or equal to your installed CUDA version.

> **Important:** Do not commit `config.sh` to version control, as it contains
> your Hugging Face access token. It is listed in `.gitignore` for this reason.

### Step 2: Create the Conda Environment

Submit the environment creation job:

```bash
sbatch < create_environment.sh
```

This job installs PyTorch, the Hugging Face Transformers library, BitsAndBytes,
and other dependencies into a new conda environment. It may take several minutes
to complete.

### Step 3: Download Models

The models to download are listed in `model_list.txt`, one model ID per line:

```
meta-llama/Meta-Llama-3.1-8B-Instruct
google/gemma-4-12B-it
Qwen/Qwen3.5-9B
```

Edit this file to add or remove models as needed, then submit the download job:

```bash
sbatch < download_models.sh
```

This job downloads model weights to `$HF_HOME/hub/`. Model sizes vary -- allow
sufficient storage space (roughly 15-20 GB per model at full precision). A GPU
node is not required for this step.

---

## Running the Example

The example script (`run_example.py`) uses `meta-llama/Meta-Llama-3.1-8B-Instruct`
to classify the article in `example_text.md` as describing a flood or hurricane
disaster event or not.

Submit the job to the [l40-gpu](https://help.rc.unc.edu/gpu/) partition with:

```bash
sbatch < run_example.sh
```

Expected output (see `example-slurm.out` for a full example):

```
Loading model: meta-llama/Meta-Llama-3.1-8B-Instruct
  Model loaded. Parameters are on device: cuda:0
  Generating response...

==============PROMPT==============

Does the following news article describe an actual flood or hurricane natural disaster event?
Answer YES or NO on the first line. On the second line, give a brief one-sentence explanation.
Article: example_text.md

==========MODEL RESPONSE==========

YES
The article describes a flood event caused by Tropical Storm Chantal, which
brought heavy rainfall and resulted in significant flooding in the Eastgate
Crossing shopping center in Chapel Hill.
```

Due to the high demand for GPU sessions on Longleaf, please familiarize yourself with how to monitor your code's usage of allocated [GPU resources](https://help.rc.unc.edu/gpu/) and avoid requesting more than you need. 

---

## Adapting This Example to Your Own Use Case

To use a different model, update the `model_id` variable at the top of
`run_example.py` to any model ID listed in `model_list.txt`.

To classify a different article, replace `example_text.md` with your own
text file and update the `input_text_path` variable in `run_example.py`.

To adapt the classification task itself, modify the `messages` list in
`run_example.py` to change the system prompt or user prompt.

---

## Acknowledgements

The code in this repository was developed with assistance from
[Claude](https://www.anthropic.com/claude) (Anthropic), an AI assistant
available to UNC researchers through [PromptLab](https://promptlab.lib.unc.edu/),
a service of the UNC University Library.

The authors also gratefully acknowledge [UNC ITS Research Computing](https://help.rc.unc.edu)
for providing and maintaining the computational infrastructure -- including the
Longleaf HPC cluster and its GPU resources -- that makes running large language
models locally possible.

---

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md)
for details.

