from samplePositional import get_positional_encoding

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np

# =========================
# 1. Define Label Constants
# =========================
WALKING = 0
RUNNING = 1
DANCING = 2
EATING = 3
SLEEPING = 4
CODING = 5

# =========================
# 2. Updated Vocabulary
# =========================
vocab = {
    "<PAD>": 0, "i": 1, "am": 2, "walking": 3, "running": 4, 
    "dancing": 5, "eating": 6, "sleeping": 7, "coding": 8, 
    "walk": 9, "run": 10, "dance": 11, "eat": 12, "sleep": 13, 
    "code": 14, "walks": 15, "runs": 16, "dances": 17, 
    "eats": 18, "sleeps": 19, "codes": 20
}
vocab_size = len(vocab)

# =====================================
# 3. Hyperparameters and Label Mapping
# =====================================
D_MODEL = 128           # Increased Embedding dimension from 16 to 128
NUM_CLASSES = 6         # Updated to match the number of label constants
MAX_LEN = 10           
BATCH_SIZE = 8          # Increased batch size for better gradient estimates
EPOCHS = 20
LR = 0.001

# =====================================
# 4. Expanded Dataset with Constants
# =====================================
sentences = [
    # Walking
    "i am walking", "i walk every morning", "walking am i", 
    "she walks to school", "walking is fun", "walks am i",

    # Running
    "i am running", "i run in the park", "running am i", 
    "he runs daily", "running keeps me fit", "runs am i",

    # Dancing
    "i am dancing", "i dance at parties", "dancing am i", 
    "she dances gracefully", "dancing is joyful", "dances am i",

    # Eating
    "i am eating", "i eat an apple", "eating am i",
    "he eats quickly", "eating makes me happy", "eats am i",

    # Sleeping
    "i am sleeping", "i sleep early", "sleeping am i", 
    "she sleeps peacefully", "sleeping is essential", "sleeps am i",

    # Coding
    "i am coding", "i code in Python", "coding am i", 
    "he codes efficiently", "coding challenges me", "codes am i"
]

labels = [
    # Walking
    WALKING, WALKING, WALKING, 
    WALKING, WALKING, WALKING,

    # Running
    RUNNING, RUNNING, RUNNING, 
    RUNNING, RUNNING, RUNNING,

    # Dancing
    DANCING, DANCING, DANCING, 
    DANCING, DANCING, DANCING,

    # Eating
    EATING, EATING, EATING, 
    EATING, EATING, EATING,

    # Sleeping
    SLEEPING, SLEEPING, SLEEPING, 
    SLEEPING, SLEEPING, SLEEPING,

    # Coding
    CODING, CODING, CODING, 
    CODING, CODING, CODING
]  # Updated labels using constants

# =========================
# 5. Tokenizer Function
# =========================
def tokenize(sentence):
    return [vocab.get(word, vocab["<PAD>"]) for word in sentence.lower().split()]

# =========================
# 6. Dataset Class
# =========================
class ActionDataset(Dataset):
    def __init__(self, sentences, labels, max_len):
        self.sentences = sentences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = tokenize(self.sentences[idx])
        # Padding
        if len(tokens) < self.max_len:
            tokens += [vocab["<PAD>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# =========================
# 7. Transformer Model
# =========================
class BasicTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, max_len):
        super(BasicTransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = torch.tensor(get_positional_encoding(max_len, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=4, 
                dim_feedforward=256, 
                batch_first=True  # Set batch_first to True
            ),
            num_layers=4
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:x.size(1), :].to(x.device)
        transformed = self.transformer_encoder(embedded)
        out = self.fc(transformed.mean(dim=1))  # Average pooling
        return out

# =========================
# 8. Prepare Data Loaders
# =========================
train_dataset = ActionDataset(sentences, labels, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# 9. Initialize Model, Loss, Optimizer
# =========================
model = BasicTransformerClassifier(vocab_size=vocab_size, d_model=D_MODEL, num_classes=NUM_CLASSES, max_len=MAX_LEN)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# 10. Training Loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for tokens, label in train_loader:
        optimizer.zero_grad()
        output = model(tokens)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

print("Training complete.")

# =========================
# 11. Inference with Constants
# =========================
test_sentence = "i am dancing"
test_tokens = tokenize(test_sentence)
if len(test_tokens) < MAX_LEN:
    test_tokens += [vocab["<PAD>"]] * (MAX_LEN - len(test_tokens))
else:
    test_tokens = test_tokens[:MAX_LEN]
test_tensor = torch.tensor(test_tokens, dtype=torch.long).unsqueeze(0)

# Define action mapping using constants
action_map = {
    WALKING: "walking",
    RUNNING: "running",
    DANCING: "dancing",
    EATING: "eating",
    SLEEPING: "sleeping",
    CODING: "coding"
}

# Inference
model.eval()
with torch.no_grad():
    output = model(test_tensor)
    predicted_action = torch.argmax(output, dim=1).item()
    print(f"Predicted action for '{test_sentence}': {action_map.get(predicted_action, 'Unknown')}")
