import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import os

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    # Load model in bfloat16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Load the tokenizer from the original model
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=2048):
    """Generate text from the model with streaming output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Setup streamer for token-by-token generation
    streamer = TextIteratorStreamer(tokenizer)
    
    # Generate text in a separate thread
    generation_kwargs = dict(
        inputs=inputs["input_ids"],
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        streamer=streamer,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Print tokens as they're generated
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text
    
    return generated_text

def main():
    # Path to your fine-tuned model (current directory where safetensor files are)
    model_path = "."
    
    if not os.path.exists("model.safetensors.index.json"):
        print("Error: Model files not found in current directory")
        return
    
    print("Loading model...")
    model, tokenizer = load_model(model_path)
    print("Model loaded successfully!")
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        print("\nGenerating response...\n")
        generate_text(model, tokenizer, prompt)

if __name__ == "__main__":
    main() 