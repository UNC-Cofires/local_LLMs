import numpy as np
from huggingface_hub import snapshot_download

# Get list of models to download
models = np.loadtxt('model_list.txt',dtype=str)

# Download each model in list
for model_id in models:
    print(f"Downloading {model_id}...")
    snapshot_download(model_id)
    print(f"  Done: {model_id}")

print("All models downloaded.")
