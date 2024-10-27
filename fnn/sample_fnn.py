import torch
import torch.nn as nn
import openai

# Function to obtain gpt embeddings
openai.api_key = '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' # Get openAI API Key

def get_gpt_embeddings(user_question):
    response = openai.Embedding.create(
        input=user_question,
        model="text-embedding-ada-002"
    )
    
    embeddings = response['data'][0]['embedding']
    return torch.tensor(embeddings).unsqueeze(0)

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
    
d_model = 512 
d_ff = 2048 

ffn = FeedForwardNetwork(d_model, d_ff)

batch_size = 2   # Number of sequences in a batch
seq_len = 10     # Length of each sequence (number of tokens)

user_question = input("Input Astronaut Question: ")
# x = torch.randn(batch_size, seq_len, d_model)
x = get_gpt_embeddings(user_question)

# Pass the input through the feedforward network
output = ffn(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output tensor:", output)
