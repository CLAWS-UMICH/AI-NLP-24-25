from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Define a function for generating responses with the Hugging Face model
def generate_response(prompt):
    # Define generation parameters
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # Set this to the desired number of tokens
        pad_token_id=tokenizer.eos_token_id  # Set padding token ID to EOS token ID
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create a prompt template
prompt_template = PromptTemplate(input_variables=["query"], template="{query}")

# Define the response function
def langchain_agent_response(query: str):
    # Render the prompt from the template
    prompt_text = prompt_template.format(query=query)
    # Generate the response using the model
    response = generate_response(prompt_text)
    return response

# Example usage
if __name__ == "__main__":
    prompt = input("Astronaut prompt: ")
    print(langchain_agent_response(prompt))