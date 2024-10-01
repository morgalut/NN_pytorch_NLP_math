import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from model import SimpleNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def arithmetic_loss(predictions, targets):
    """
    Custom arithmetic loss that calculates the difference between
    predictions and targets using mean absolute error.
    """
    # You can use Mean Absolute Error (MAE) as a base for arithmetic-like loss
    loss = torch.mean(torch.abs(predictions - targets))
    return loss

def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), '..', 'data', filename)

def clean_text(text):
    """ Remove unwanted characters from text. """
    return pd.Series(text).replace(r"[\\.,/]", '', regex=True).iloc[0] if isinstance(text, str) else text

def load_and_clean_data(file_name, na_threshold=0.4):
    """ Load, clean, and prepare dataset for training. """
    file_path = get_data_path(file_name)
    try:
        # Use `on_bad_lines='skip'` to skip rows with inconsistent columns
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        if df.empty: 
            raise ValueError("Empty dataset.")
        
        # Clean and drop rows with too many NaN values
        df_cleaned = df.dropna(thresh=int((1 - na_threshold) * df.shape[1]))
        df_cleaned = df_cleaned.applymap(clean_text).apply(pd.to_numeric, errors='coerce').dropna()

        if df_cleaned.empty: 
            raise ValueError("Empty after cleaning.")
        return df_cleaned

    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        raise

def convert_to_tensor(df):
    return torch.tensor(df.astype(np.float32).values, dtype=torch.float32).to(device)

class Trainer:
    def __init__(self, model, criterion, optimizer, batch_size=32, num_epochs=10, use_amp=False):
        self.model, self.criterion, self.optimizer = model.to(device), criterion, optimizer
        self.batch_size, self.num_epochs, self.use_amp = batch_size, num_epochs, use_amp
        self.scaler = GradScaler() if use_amp else None

    def create_dataloader(self, embeddings, labels):
        return DataLoader(TensorDataset(embeddings, labels), batch_size=self.batch_size, shuffle=True)

    def train(self, train_embeddings, train_labels):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for X, y in self.create_dataloader(train_embeddings, train_labels):
                self.optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                (self.scaler.scale(loss) if self.use_amp else loss).backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss:.4f}")

    def evaluate(self, test_embeddings, test_labels):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.create_dataloader(test_embeddings, test_labels):
                predicted = torch.argmax(self.model(X), dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        print(f"Accuracy: {100 * correct / total:.2f}%")

    @staticmethod
    def save_model(model, optimizer, epoch, path):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, path)

    @staticmethod
    def load_model(model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']
    
if __name__ == "__main__":
    train_data = load_and_clean_data("train.csv")
    train_embeddings, train_labels = convert_to_tensor(train_data.iloc[:, :-1]), convert_to_tensor(train_data.iloc[:, -1])

    test_data = load_and_clean_data("test.csv")
    test_embeddings, test_labels = convert_to_tensor(test_data.iloc[:, :-1]), convert_to_tensor(test_data.iloc[:, -1])

    input_size, output_size = train_embeddings.shape[1], len(torch.unique(train_labels))
    model = SimpleNN(output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, nn.CrossEntropyLoss(), optimizer, batch_size=32, num_epochs=10, use_amp=True)
    trainer.train(train_embeddings, train_labels)
    trainer.evaluate(test_embeddings, test_labels)
    trainer.save_model(model, optimizer, trainer.num_epochs, "model_checkpoint.pth")
