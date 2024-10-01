import os
import torch
import pandas as pd
from data_preprocessing import DataPreprocessor
from embedding_generation import EmbeddingGenerator
from Enum.data_type import DataType
from training import Trainer, arithmetic_loss
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from model import SimpleNN
from nlp_analysis import ArithmeticNLP  # Import the NLP analysis module

def load_csv_data(file_path):
    """
    Load CSV data into a Pandas DataFrame, skipping bad lines.
    """
    try:
        # Skip rows that have incorrect number of columns
        return pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nlp_analyzer = ArithmeticNLP()

    # Define file paths
    file_paths = {
        DataType.TRAIN.value: 'data/train.csv',
        DataType.TEST.value: 'data/test.csv',
        DataType.MISCONCEPTION.value: 'data/misconception_mapping.csv'
    }

    # Load data using Pandas
    train_df = load_csv_data(file_paths[DataType.TRAIN.value])
    test_df = load_csv_data(file_paths[DataType.TEST.value])

    if train_df is None or test_df is None:
        print("Error: Required data not loaded.")
        return

    # Initialize your data preprocessor
    preprocessor = DataPreprocessor()

    # Preprocess the text using Pandas
    train_df['cleaned_text'] = train_df['QuestionText'].apply(nlp_analyzer.preprocess_text)
    test_df['cleaned_text'] = test_df['QuestionText'].apply(nlp_analyzer.preprocess_text)

    # Ensure the 'CorrectAnswer' column is renamed to 'labels'
    train_df.rename(columns={'CorrectAnswer': 'labels'}, inplace=True)
    test_df.rename(columns={'CorrectAnswer': 'labels'}, inplace=True)

    # Generate embeddings using the embedding generator
    embedding_generator = EmbeddingGenerator('distilbert-base-uncased')
    train_embeddings = torch.tensor(embedding_generator.generate_embeddings(train_df['cleaned_text'].tolist()), dtype=torch.float32).to(device)
    test_embeddings = torch.tensor(embedding_generator.generate_embeddings(test_df['cleaned_text'].tolist()), dtype=torch.float32).to(device)

    # Encode labels using scikit-learn
    label_encoder = LabelEncoder()
    train_labels = torch.tensor(label_encoder.fit_transform(train_df['labels']), dtype=torch.long).to(device)
    test_labels = torch.tensor(label_encoder.transform(test_df['labels']), dtype=torch.long).to(device)

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(num_classes=4).to(device)  # Send model to the device (GPU or CPU)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model using PyTorch
    trainer = Trainer(model, criterion, optimizer, batch_size=32, num_epochs=10, use_amp=torch.cuda.is_available())
    trainer.train(train_embeddings, train_labels)

    # Evaluate the model on the test data
    trainer.evaluate(test_embeddings, test_labels)


if __name__ == "__main__":
    main()
