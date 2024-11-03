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

# =====================================
# Global Configuration
# =====================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================
# Hyperparameters
# =====================================
D_MODEL = 512
MAX_LEN = 10
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.0003

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
class BasicTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, max_len):
        super(BasicTransformerClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Convert positional encoding from NumPy to Torch Tensor and register as buffer
        pos_encoder_np = get_positional_encoding(max_len, d_model)
        pos_encoder_tensor = torch.from_numpy(pos_encoder_np).float()  # Convert to tensor
        pos_encoder_tensor = pos_encoder_tensor.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pos_encoder', pos_encoder_tensor)  # Register as buffer
        
        # Create transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        # Stack multiple transformer layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6  
        )
        
        # Simple but effective classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_len)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Embedding with positional encoding
        x = self.embedding(x) + self.pos_encoder  # Broadcasting over batch dimension
        
        # Apply transformer stack
        x = self.transformer(x)
        
        # Global max pooling instead of mean
        x = torch.max(x, dim=1)[0]
        
        # Classification
        return self.classifier(x)

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

    # Generate data and build vocabulary first
    synthetic_data = generate_synthetic_data(num_samples=10000)
    vocab = build_vocabulary(synthetic_data)  # Get vocab from actual data
    
    model = BasicTransformerClassifier(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        num_classes=NUM_CLASSES,
        max_len=MAX_LEN
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
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
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
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    val_loss += loss.item()
                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.numel()
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
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