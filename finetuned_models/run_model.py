#https://drive.google.com/file/d/1vzPZqeQTesCzhs2hSBzus8RGDv6ppbcT/view?usp=sharing

from llama_cpp import Llama

# Initialize the Llama model with correct parameters
llm = Llama(
    model_path="./screen_selecting.gguf",  # Path to your GGUF file
    n_ctx=8192,                # Match your max_seq_length
    n_threads=8,               # Use all CPU cores
    n_gpu_layers=0,            # 0 for CPU-only
    verbose=False,
)

# Create prompt using your original template
prompt = "show my vitals"

template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{0}


### Input:
{1}

### Response:
{2}"""


# Generate response with proper configuration and streaming
output = llm.create_chat_completion(
    messages=[{"role": "user", "content": template.format(prompt, "", "")}],
    temperature=0,
    max_tokens=128,
    stream=True
)

# Print streamed output
for chunk in output:
    if chunk['choices'][0]['delta'].get('content'):
        print(chunk['choices'][0]['delta']['content'], end='', flush=True)
print()  # New line at end

def generate_response(prompt):
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": template.format(prompt, "", "")}],
        temperature=0,
        max_tokens=128,
    )
    return output['choices'][0]['message']['content']
