import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        # Linear layers to transform Q, K, V
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, input_dim)
        """
        # Step 1: Compute Q, K, V
        Q = self.query_layer(x)  # (batch_size, seq_length, input_dim)
        K = self.key_layer(x)     # (batch_size, seq_length, input_dim)
        V = self.value_layer(x)   # (batch_size, seq_length, input_dim)
        
        # Step 2: Compute attention scores as Q @ K^T (transpose of K)
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_length, seq_length)
        
        # Step 3: Normalize attention scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_length, seq_length)
        
        # Step 4: Compute weighted sum of values
        output = torch.bmm(attn_weights, V)  # (batch_size, seq_length, input_dim)
        
        return output, attn_weights

# Example usage
batch_size = 2
seq_length = 5
input_dim = 4

# Create a batch of input sequences
x = torch.rand(batch_size, seq_length, input_dim)

# Initialize and apply attention layer
attention_layer = SimpleAttention(input_dim)
output, attn_weights = attention_layer(x)

print("Output shape:", output.shape)  # Expected: (batch_size, seq_length, input_dim)
print("Attention Weights shape:", attn_weights.shape)  # Expected: (batch_size, seq_length, seq_length)