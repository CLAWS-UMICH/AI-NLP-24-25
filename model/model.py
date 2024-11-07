#To train:
# python model.py --train

#To run:
# python model.py

from data import (
    vocab, tokenize, generate_synthetic_data, test_sentences, NUM_CLASSES,
    action_map_order, build_vocabulary  # Updated to use the list
)
from samplePositional import get_positional_encoding

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import string
import math

# =====================================
# Global Configuration
# =====================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================
# Hyperparameters
# =====================================
D_MODEL = 256
MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
NUM_HEADS = 8
NUM_LAYERS = 3

# =========================
# Dataset Class
# =========================
class ActionDataset(Dataset):
    def __init__(self, data, max_len):
        self.sentences = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = [vocab.get(word.strip(string.punctuation).lower(), vocab["<PAD>"]) 
                 for word in self.sentences[idx].split()]
        
        if len(tokens) < self.max_len:
            tokens += [vocab["<PAD>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        label_vector = [((self.labels[idx] >> i) & 1) for i in range(NUM_CLASSES)]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label_vector, dtype=torch.float)

# =========================
# Model Definition
# =========================
class ImprovedTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, 
                 max_len=64, num_classes=NUM_CLASSES, dropout=0.1):
        super(ImprovedTransformerClassifier, self).__init__()
        
        self.embed_dim = embed_dim  # Store embed_dim as instance variable
        self.max_len = max_len      # Store max_len as instance variable
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.embed_dropout = nn.Dropout(dropout)
        
        # Initialize position embeddings
        self.init_pos_embeddings()
        
        # Use PyTorch's built-in transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def init_pos_embeddings(self):
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(self.max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:self.embed_dim//2])  # Make sure we don't overflow
        self.pos_embedding.data = pe.unsqueeze(0)

    def forward(self, x):
        # Create padding mask
        padding_mask = (x == vocab["<PAD>"]).to(x.device)
        
        # Embedding layer
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.embed_dropout(x)
        
        # Transformer layers
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling (excluding padding)
        mask_expanded = ~padding_mask.unsqueeze(-1).expand(x.size())
        sum_embeddings = (x * mask_expanded.float()).sum(dim=1)
        count_tokens = mask_expanded.float().sum(dim=1)
        pooled_output = sum_embeddings / count_tokens.clamp(min=1e-9)
        
        # Classification
        return self.classifier(pooled_output)

# =========================
# Action Prediction Function
# =========================
def predict_action(model, sentence, max_len=MAX_LEN, threshold=0.5):
    model.eval()
    with torch.no_grad():
        test_tokens = tokenize(sentence)
        if len(test_tokens) < max_len:
            test_tokens += [vocab["<PAD>"]] * (max_len - len(test_tokens))
        else:
            test_tokens = test_tokens[:max_len]
        
        test_tensor = torch.tensor(test_tokens, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(test_tensor)
        probabilities = torch.sigmoid(logits)[0]
        
        print("\nPredicted actions:")
        detected_actions = []
        for i, action_name in enumerate(action_map_order):
            prob = probabilities[i].item() * 100
            print(f"{action_name:10}: {prob:.1f}%")
            if prob > threshold * 100:
                detected_actions.append(action_name)
        
        return ", ".join(detected_actions) if detected_actions else "No actions detected"

# =========================
# Main Function
# =========================
def main():
    parser = argparse.ArgumentParser(description='Predict action from input sentence')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--sentence', type=str, help='Input sentence to evaluate')
    args = parser.parse_args()

    # Move data generation before the vocabulary building
    synthetic_data = generate_synthetic_data(num_samples=10000)
    
    # Load the model first to ensure we use the same vocabulary size
    try:
        # Try to load the model to get the vocab size
        state_dict = torch.load('best_model.pth')
        vocab_size = state_dict['embedding.weight'].shape[0]
        model = ImprovedTransformerClassifier(
            vocab_size=vocab_size,
            embed_dim=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            max_len=MAX_LEN,
            num_classes=NUM_CLASSES,
            dropout=0.1
        ).to(device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        # If no model exists, build vocabulary and create new model
        vocab = build_vocabulary(synthetic_data)
        model = ImprovedTransformerClassifier(
            vocab_size=len(vocab),
            embed_dim=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            max_len=MAX_LEN,
            num_classes=NUM_CLASSES,
            dropout=0.1
        ).to(device)

    if args.train:
        # No need for separate validation data generation
        train_dataset = ActionDataset(synthetic_data[:9000], MAX_LEN)  # Use 90% for training
        val_dataset = ActionDataset(synthetic_data[9000:], MAX_LEN)    # Use 10% for validation
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Use weighted loss
        pos_weight = torch.ones(NUM_CLASSES).to(device) * 2.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Use AdamW with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR * 10,
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        best_val_acc = 0
        patience = 10
        no_improve = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Add progress tracking
            total_batches = len(train_loader)
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("Training Progress:")
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.numel()
                
                # Print progress every 10% of batches
                if (batch_idx + 1) % (total_batches // 10) == 0:
                    current_loss = train_loss / (batch_idx + 1)
                    current_acc = train_correct / train_total
                    progress = (batch_idx + 1) / total_batches * 100
                    print(f"Progress: {progress:.1f}% | Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
            
            # Validation phase with progress tracking
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            print("\nValidation Progress:")
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    val_loss += loss.item()
                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.numel()
                    
                    # Print progress every 25% of validation
                    if (batch_idx + 1) % (len(val_loader) // 4) == 0:
                        current_loss = val_loss / (batch_idx + 1)
                        current_acc = val_correct / val_total
                        progress = (batch_idx + 1) / len(val_loader) * 100
                        print(f"Progress: {progress:.1f}% | Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
            
            # End of epoch summary
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            print("\nEpoch Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Sample predictions every 5 epochs
            if (epoch + 1) % 5 == 0:
                print("\nSample Predictions:")
                model.eval()
                with torch.no_grad():
                    for test_sent in test_sentences[:3]:  # Only show 3 examples
                        pred = predict_action(model, test_sent)
                        print(f"Input: '{test_sent}'")
                        print(f"Prediction: {pred}\n")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Test on sample sentences every 10 epochs
            if (epoch + 1) % 10 == 0:
                print("\nTesting sample sentences:")
                for test_sent in test_sentences:
                    pred = predict_action(model, test_sent)
                    print(f"'{test_sent}' -> {pred}")
                print()
                
        print("Training completed.")
    else:
        try:
            model.load_state_dict(torch.load('best_model.pth'))
        except FileNotFoundError:
            print("Error: Model file not found. Please train the model first using --train")
            return

    if args.sentence:
        prediction = predict_action(model, args.sentence)
        print(f"Predicted action for '{args.sentence}': {prediction}")
    elif not args.train:
        print("Enter sentences to predict actions (type 'quit' to exit):")
        while True:
            sentence = input("> ")
            if sentence.lower() == 'quit':
                break
            prediction = predict_action(model, sentence)
            print(f"Predicted action: {prediction}")

if __name__ == "__main__":
    main()