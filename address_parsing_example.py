import numpy as np
import pandas as pd
import time
from datetime import timedelta
from copy import deepcopy
import json
import sys
import os
from vllm import LLM, SamplingParams
from address_parsing_prompts import build_prompt

###----------FUNCTION DEFINITIONS----------###
# Function definitions are safe at the top level — they are not executed
# at import time, only defined. They reference `llm` and `sampling_params`
# as globals, which will be defined inside the __main__ block before
# any of these functions are called.

def _format_prompt(messages: list[dict]) -> str:
    """
    Converts a list of chat message dicts into a single formatted string.

    vLLM's generate() method expects raw text strings, not message dicts.
    apply_chat_template() handles converting the system/user/assistant turns
    into whatever input format the specific model expects (e.g. Gemma's
    <start_of_turn> / <end_of_turn> tokens).

    Note: if llm.get_tokenizer() is not available in your vLLM version,
    you can load the tokenizer separately instead:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    """
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages,
        # tokenize=False: Return a string rather than token IDs.
        # vLLM handles tokenization internally — we just need the string.
        tokenize=False,
        # add_generation_prompt=True: Appends the model's "start of response"
        # marker so the model knows it should begin generating a reply.
        add_generation_prompt=True,
        # enable_thinkign=False: disable thinking mode to save on computation
        # this means that the model provides only the final response without
        # explaining its reasoning. 
        enable_thinking=False,
    )

    return prompt

def parse_addresses_batch(addresses: list[str]) -> list[str]:
    """
    Parse a list of address strings in a single vLLM call.

    vLLM's core advantage over standard transformers inference is its ability
    to efficiently schedule many requests concurrently using PagedAttention —
    a memory management technique that avoids wasting GPU memory on padding.
    Passing all addresses at once maximizes GPU utilization and throughput
    compared to calling parse_address() in a loop.

    Returns a list of raw JSON strings in the same order as the input list.
    """
    # Build and format a prompt string for each address
    prompts = [
        _format_prompt(build_prompt(address))
        for address in addresses
    ]

    # vLLM schedules all prompts together and returns results in input order
    outputs = llm.generate(prompts, sampling_params)

    # Extract the generated text string from each RequestOutput object
    return [out.outputs[0].text for out in outputs]

def postprocess_output_text(result,address_string,address_id):
    """
    This function checks whether the string representation of address components
    returned by an LLM is a valid JSON  while adding fields for the address_id 
    and input address_string. 
    
    Returns a python dictionary. 
    """

    address_dict = {'address_id':address_id,'address_string':address_string,'parsing_errors':False}

    # Attempt to parse JSON output
    try:
        address_dict['address_components'] = json.dumps(json.loads(result),indent=4)

    # If errors occur, make a note of this and return an empty JSON array
    except:
        address_dict['parsing_errors'] = True
        address_dict['address_components'] = json.dumps([],indent=4)

    return address_dict

###--------------MAIN EXECUTION--------------###
# Everything that spawns subprocesses — including LLM() — must be inside this guard.
# When vLLM spawns a subprocess and re-imports this script, the subprocess
# will skip this block entirely, preventing the infinite spawn loop.

if __name__ == '__main__':

    ### *** INITIAL SETUP *** ###
    
    # Get current working directory
    pwd = os.getcwd()
    
    # Get ID of local LLM to run
    # (passed as a command-line argument to the Python script)
    MODEL_ID = sys.argv[1] 
    MODEL_NAME = MODEL_ID.split('/')[-1]
    print(f'\nLocal LLM: {MODEL_NAME}\n')
    
    # Load address data
    address_data = pd.read_csv('address_parsing_input_data.csv')
    
    ### *** MODEL LOADING *** ###
    
    # LLM() starts vLLM's inference engine and loads the model into GPU memory.
    # This happens once at startup. 
    
    llm = LLM(
        
        model=MODEL_ID,
    
        # dtype: Numeric precision for model weights.
        # "bfloat16" halves memory usage vs. float32 with minimal quality loss.
        dtype="bfloat16",
    
        # max_model_len: Maximum total sequence length (prompt + generated tokens).
        # vLLM pre-allocates its KV cache based on this value, so keeping it
        # smaller frees memory for larger batch sizes.
        # The few-shot prompt in this script is roughly 8000 tokens;
        # 16000 gives ample headroom. Increase this value if you hit
        # context-length errors at runtime.
        max_model_len=16000,
    
        # gpu_memory_utilization: Fraction of GPU VRAM vLLM may use (0.0–1.0).
        # After loading model weights, vLLM pre-allocates the remaining share
        # of this budget for the KV cache. 0.90 leaves a small safety margin
        # to avoid out-of-memory errors from other GPU overhead.
        gpu_memory_utilization=0.90,
    )
    
    ### *** SAMPLING PARAMETERS *** ###
    
    # Controls how the model generates output tokens.
    
    sampling_params = SamplingParams(
        # temperature: Controls randomness in token selection.
        # 0.0 = greedy decoding (always pick the highest-probability token).
        # This is ideal for structured output tasks like JSON generation,
        # where you want deterministic, consistent results rather than variety.
        temperature=0.0,
    
        # max_tokens: Maximum number of new tokens to generate per request.
        # Setting this unnecessarily high wastes compute on padding;
        # setting it too low risks truncating valid output.
        max_tokens=2048,
    )
    
    ### *** EXTRACT ADDRESS COMPONENTS FROM RAW STRINGS *** ###
    
    
    # Get list of addresses to process in this chunk
    address_ids = address_data['id'].tolist()
    address_strings = address_data['address'].tolist()

    # Parse addresses
    results = parse_addresses_batch(address_strings)
    results_df = pd.DataFrame([postprocess_output_text(result,address_string,address_id) for result,address_string,address_id in zip(results,address_strings,address_ids)])

    # Record the specific LLM used for parsing
    # (Useful if you want to check for consistency of output across models)
    results_df['model_id'] = MODEL_ID

    # Save results
    results_df.to_csv('address_parsing_output_data.csv',index=False)
