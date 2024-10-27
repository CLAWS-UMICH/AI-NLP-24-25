import torch
import torch.nn as nn
import openai

# Function to obtain gpt embeddings
client = openai.OpenAI(api_key='') # Get openAI API Key

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return torch.tensor(response['data'][0]['embedding']).unsqueeze(0)

class FeedForwardNetwork(nn.Module):
    # d_model is dimensions of input embeddings
    # d_ff is dimensions of hidden layer in FNN, d_ff > d_model
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        # 1st linear layer transforms input from d_model to d_ff dimensions
        self.linear1 = nn.Linear(d_model, d_ff)
        # Activation function, could be another one like leaky ReLU
        self.relu = nn.ReLU()
        # Second linear layer transforms back from d_ff to d_model dimensions
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # Apply first linear transformation
        out = self.linear1(x)
        # Apply ReLU activation
        out = self.relu(out)
        # Apply second linear transformation
        out = self.linear2(out)
        return out
    
d_model = 1536
d_ff = 4096 

ffn = FeedForwardNetwork(d_model, d_ff)

def process_user_input(user_input):
    embedding = get_embedding(user_input)
    ffn_output = ffn(embedding)

    print("FFN Output Shape:", ffn_output.shape)
    print("FFN Output Tensor:", ffn_output)

    return ffn_output

# Example usage
user_question = input("Ask me a question: ")
ffn_result = process_user_input(user_question)
