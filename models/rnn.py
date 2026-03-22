import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm

from models.base_model import BaseEmotionModel
from data.data_loader import get_dataloader, preprocess_text, LABEL_NAMES
from results import exporter


class RNN(nn.Module):
    """A Bidirectional LSTM for text classification."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int, padding_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


class RNNModel(BaseEmotionModel):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def train(self, train_data, val_data, epochs=10, batch_size=64) -> None:
        self.model = RNN(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=128,
            hidden_dim=128,
            output_dim=len(LABEL_NAMES),
            padding_idx=self.tokenizer.pad_token_id,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            train_loader = list(get_dataloader(train_data, batch_size, self.tokenizer))
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                predictions = self.model(input_ids)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_results = self._evaluate_internal(val_data, criterion, batch_size)
            
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(epoch_loss / len(train_loader))
            self.history["val_loss"].append(val_results["loss"])
            self.history["val_accuracy"].append(val_results["accuracy"])

            print(f"  Train Loss: {self.history['train_loss'][-1]:.4f} | "
                  f"Val Loss: {val_results['loss']:.4f} | "
                  f"Val Acc: {val_results['accuracy']:.4f}")

    def predict(self, texts: list[str]) -> list[int]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()
        processed = [preprocess_text(t) for t in texts]
        
        all_preds = []
        batch_size = 128
        for i in range(0, len(processed), batch_size):
            batch_texts = processed[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs["input_ids"])
                preds = torch.argmax(outputs, dim=1).tolist()
                all_preds.extend(preds)
        
        return all_preds

    def evaluate(self, test_data) -> dict:
        y_true = test_data["label"]
        y_pred = self.predict(test_data["text"])
        metrics = self.get_metrics(y_true, y_pred)
        
        print(f"Test Results — RNN (LSTM):")
        for k, v in metrics.items():
            print(f"  {k:<10}: {v:.4f}")
            
        exporter.save_model_metrics_bar(metrics, "rnn")
        exporter.save_text_summary(metrics, {"type": "Bi-LSTM", "hidden_dim": 128}, "rnn")
        
        return metrics

    def _evaluate_internal(self, data, criterion, batch_size):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        val_loader = list(get_dataloader(data, batch_size, self.tokenizer))
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total,
        }
