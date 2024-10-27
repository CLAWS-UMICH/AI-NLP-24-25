import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Model dimensions
        self.d_model = 128
        self.sequence_length = 4
        
        # Positional encoding
        position = torch.arange(self.sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, self.sequence_length, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Layers
        self.input_layer = nn.Linear(1, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,  # Increased for more capacity
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3  # Increased number of layers
        )
        
        self.output_layer = nn.Linear(self.d_model, 1)
        
    def forward(self, x):
        # Input embedding + positional encoding
        x = self.input_layer(x)
        x = x + self.pe
        
        # Transformer processing
        x = self.transformer(x)
        
        # Output
        return self.output_layer(x)

def generate_sequences(num_batches, batch_size, seq_length):
    sequences = []
    targets = []
    
    for _ in range(num_batches):
        batch_sequences = []
        batch_targets = []
        
        for _ in range(batch_size):
            # Start with 1
            seq = [1]
            
            # Each number is the sum of all previous numbers
            for i in range(seq_length - 1):
                next_num = sum(seq)
                seq.append(next_num)
            
            # Input sequence is all but last number
            batch_sequences.append(seq[:-1])
            # Target is the last number
            batch_targets.append(seq[-1])
            
        sequences.append(torch.tensor(batch_sequences, dtype=torch.float32))
        targets.append(torch.tensor(batch_targets, dtype=torch.float32))
    
    return torch.stack(sequences), torch.stack(targets)

# Training parameters
sequence_length = 5  # 4 input numbers + 1 target
batch_size = 32
num_batches = 80
num_epochs = 75
learning_rate = 0.0005  # Reduced learning rate for stability

# Generate training data
x_data, y_data = generate_sequences(num_batches, batch_size, sequence_length)
x_data = x_data.unsqueeze(-1)  # Add feature dimension
y_data = y_data.unsqueeze(-1)  # Add feature dimension

# Initialize model and optimizer
model = SimpleTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Training loop
print("Training starting...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in range(num_batches):
        x = x_data[batch]
        y = y_data[batch]
        
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred[:, -1, :], y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

# Test the model
model.eval()
with torch.no_grad():
    # Test with our pattern sequence [1,1,2,4] → should predict 8
    test_sequence = torch.tensor([[1.0, 1.0, 2.0, 4.0]]).reshape(1, 4, 1)
    prediction = model(test_sequence)
    
    print("\nTest Results:")
    print("Input sequence:", test_sequence.reshape(-1).tolist())
    print("Predicted next number:", prediction[:, -1, :].item())
    print("Expected number:", sum(test_sequence.reshape(-1).tolist()))
    
    # Test with another sequence [4,4,8,8] → should predict 16
    test_sequence2 = torch.tensor([[4.0, 4.0, 8.0, 16.0]]).reshape(1, 4, 1)
    prediction2 = model(test_sequence2)
    
    print("\nSecond Test Results:")
    print("Input sequence:", test_sequence2.reshape(-1).tolist())
    print("Predicted next number:", prediction2[:, -1, :].item())
    print("Expected number:", sum(test_sequence2.reshape(-1).tolist()))
