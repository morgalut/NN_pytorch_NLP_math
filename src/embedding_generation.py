import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer



class EmbeddingGenerator:
    def __init__(self, model_name='distilbert-base-uncased', batch_size=16):
        self.model = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        self.model.to(self.device)
        
        
    def generate_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.dropna().tolist()  # Drop NaN values and convert to list

        # Validate and clean input texts
        cleaned_texts = [text for text in texts if isinstance(text, str) and text.strip()]

        if len(cleaned_texts) == 0:
            raise ValueError("No valid texts available for embedding generation.")

        # Tokenize and process texts in batches
        all_embeddings = []
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch_texts = cleaned_texts[i:i+self.batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = model_output.last_hidden_state.mean(dim=1)  # Average pooling
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)  # Concatenate embeddings from all batches

    def save_embeddings(self, embeddings, file_path):
        """
        Saves the embeddings to a file.

        Args:
            embeddings (torch.Tensor): The embeddings to save.
            file_path (str): The file path where the embeddings will be saved.
        """
        torch.save(embeddings, file_path)
        print(f"Embeddings saved to {file_path}")
