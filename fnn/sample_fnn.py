import torch
import torch.nn as nn

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