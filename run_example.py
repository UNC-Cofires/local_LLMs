import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

### *** CONFIGURATION *** ###

# The model ID corresponds to the model's repository on Hugging Face Hub
# (https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
# The Hugging Face library uses this ID to locate the model in your local
# cache (set by HF_HOME) without needing to re-download it each time.
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Path to the news article text file to be classified
input_text_path = "example_text.md"

### *** LOAD ARTICLE TEXT *** ###

# Read the full article text into memory as a string
with open(input_text_path,'r') as f:
    input_text = f.read()

### *** LOAD MODEL *** ###

print(f'\nLoading model: {model_id}')

# Configure 4-bit quantization using the BitsAndBytes library.
# Quantization reduces the precision of the model's weights (from 16-bit
# to 4-bit), which shrinks the model's memory footprint from ~16 GB to
# ~5-6 GB of GPU VRAM. This makes it possible to run large models on a
# single GPU. The tradeoff is a small potential reduction in output quality.
#   - load_in_4bit: enables 4-bit quantization
#   - bnb_4bit_quant_type="nf4": uses the NormalFloat4 quantization format,
#     which is recommended for LLMs
#   - bnb_4bit_compute_dtype: the dtype used for computation during the
#     forward pass; float16 balances speed and precision on modern GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16)

# Load the tokenizer for the model. The tokenizer converts raw text into
# token IDs (integers) that the model can process, and converts the model's
# output token IDs back into human-readable text.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model weights with the quantization configuration defined above.
# AutoModelForCausalLM is a generic class for text generation models.
# device_map="auto" automatically distributes the model's layers across
# available hardware (GPU, CPU, or both) based on available memory.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto")

# Confirm which device the model's parameters are loaded onto.
# next(model.parameters()) retrieves the first parameter tensor in the model,
# and .device tells us where it resides (e.g., cuda:0 for the first GPU).
# Seeing "cuda:0" here confirms the model is running on the GPU.
device = next(model.parameters()).device
print(f"  Model loaded. Parameters are on device: {device}")

### *** BUILD PROMPT *** ###

# Construct the prompt as a list of messages following the chat format
# expected by instruction-tuned models. The chat format uses alternating
# "system" and "user" (and optionally "assistant") roles:
#   - "system": sets the model's behavior and persona
#   - "user": represents the user's input or question
# This format is important because the model was fine-tuned to respond
# to this structure. Using a different format may degrade output quality.
messages = [{"role": "system",
             "content": (
                "You are a classifier that determines whether a news article "
                "describes a flood or hurricane natural disaster event. "
                "Be concise and follow the output format exactly."
             ),
            },
            {"role": "user",
             "content": (
                 "Does the following news article describe an actual flood or "
                 "hurricane natural disaster event?\n\n"
                 "Answer YES or NO on the first line. "
                 "On the second line, give a brief one-sentence explanation.\n\n"
                 f"Article:\n{input_text}"
            ),
            },
           ]

### *** TOKENIZE *** ###

# Apply the model's chat template to the messages list and tokenize the result.
# apply_chat_template() formats the messages into the specific prompt string
# that this model was trained on (e.g., wrapping messages in special tokens
# like <|begin_of_text|>, <|start_header_id|>, etc.), then converts the
# formatted string into token IDs.
#   - return_tensors="pt": return the token IDs as a PyTorch tensor
#   - add_generation_prompt=True: append a prompt marker that signals to the
#     model that it should begin generating a response
tokenized = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
)

# Extract the input_ids tensor from the tokenized output and move it to the
# same device as the model (e.g., GPU). input_ids is the sequence of integer
# token IDs that will be fed into the model.
input_ids = tokenized.input_ids.to(model.device)

### *** GENERATE RESPONSE *** ###

print("  Generating response...")

# Disable gradient computation during inference. Gradients are only needed
# during training to update model weights. Disabling them here reduces memory
# usage and speeds up inference.
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=150,   # Maximum number of new tokens to generate;
                              # limits response length
        do_sample=False,      # Use greedy decoding (always pick the most
                              # likely next token) rather than random sampling.
                              # This makes outputs deterministic, which is
                              # desirable for a classification task.
        pad_token_id=tokenizer.eos_token_id,  # Tells the model which token ID
                              # to use for padding; avoids a warning in models
                              # where the pad token is not explicitly defined
    )

# outputs[0] contains the full sequence of token IDs, including both the
# input prompt tokens and the newly generated tokens. We slice off the input
# tokens using input_ids.shape[-1] (the length of the input sequence) to
# isolate only the tokens that the model generated.
new_tokens = outputs[0][input_ids.shape[-1]:]

# Convert the generated token IDs back into a human-readable string.
# skip_special_tokens=True removes formatting tokens (e.g., <|eot_id|>)
# from the output.
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

### *** PRINT RESPONSE *** ###

# Reconstruct a readable version of the prompt for display purposes,
# replacing the full article text with just the filename to keep the
# output concise.
prompt_text = messages[1]['content'].split('Article:')[0]
prompt_text = prompt_text + f'Article: {input_text_path}'

print(f'\n==============PROMPT==============\n\n{prompt_text}')
print(f'\n==========MODEL RESPONSE==========\n\n{response}')