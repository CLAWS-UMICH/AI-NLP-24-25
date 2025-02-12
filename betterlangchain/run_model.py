from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

# Load model for CPU
model = AutoModelForCausalLM.from_pretrained("model")  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained("model")

# Format prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def get_model_response(instruction, input_text="", output=""):
    # Format the input
    prompt = alpaca_prompt.format(instruction, input_text, output)
    inputs = tokenizer([prompt], return_tensors="pt")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True
    )
    
    # Decode and extract just the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find the start of the response section
    response_start = full_response.find("### Response:") + len("### Response:")
    
    # Return just the response text, stripped of whitespace
    return full_response[response_start:].strip()

if __name__ == "__main__":
    # Example usage
    instruction = "How am I doing?"
    response = get_model_response(instruction)
    print(response)
