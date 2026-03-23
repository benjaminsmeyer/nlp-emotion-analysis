from models.base_model import BaseEmotionModel
from data.data_loader import get_dataloader, preprocess_text, LABEL_NAMES
from results import exporter

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.scale = math.sqrt(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

    def forward(self, x, pad_mask):
        normed = self.norm1(x)
        Q = self.W_q(normed)
        K = self.W_k(normed)
        V = self.W_v(normed)

        scores = (Q @ K.transpose(1, 2)) / self.scale
        scores = scores.masked_fill(pad_mask, float("-inf"))
        attention = self.dropout(torch.softmax(scores, dim=2))
        x = x + attention @ V

        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        num_classes: int,
        padding_idx: int = 0,
        num_layers: int = 4,
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embedding_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        e = self.embedding(x) + self.pos_embedding(positions)

        pad_mask = (x == self.padding_idx).unsqueeze(1)
        for layer in self.layers:
            e = layer(e, pad_mask)
        e = self.norm(e)

        token_mask = (x != self.padding_idx).unsqueeze(-1).float()
        pooled = (e * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


class TransformerModel(BaseEmotionModel):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = None
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def train(self, train_data, val_data, epochs=20, batch_size=64) -> None:
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=512,
            embedding_dim=128,
            num_classes=len(LABEL_NAMES),
            padding_idx=self.tokenizer.pad_token_id,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            train_loader = list(get_dataloader(train_data, batch_size, self.tokenizer))
            for batch in train_loader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                logits = self.model(input_ids)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            val_results = self.evaluate_validation(val_data)

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(epoch_loss / len(train_loader))
            self.history["val_loss"].append(val_results["loss"])
            self.history["val_accuracy"].append(val_results["accuracy"])

            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {self.history['train_loss'][-1]:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"Val Acc: {val_results['accuracy']:.4f}"
            )

    def predict(self, texts: list[str], batch_size=128) -> list[int]:
        if self.model is None:
            raise Exception("Transformer model cannot predict without being trained")

        self.model.eval()
        processed = [preprocess_text(t) for t in texts]
        predictions = []

        with torch.no_grad():
            for i in range(0, len(processed), batch_size):
                batch_texts = processed[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors="pt"
                )
                logits = self.model(inputs["input_ids"])
                preds = logits.argmax(dim=1).tolist()
                predictions.extend(preds)

        return predictions

    def evaluate(self, test_data) -> dict:
        y_true = test_data["label"]
        y_pred = self.predict(test_data["text"])
        metrics = self.get_metrics(y_true, y_pred)

        print("Test Results - Transformer:")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")

        exporter.save_model_metrics_bar(metrics, "transformer")
        exporter.save_text_summary(
            metrics,
            {"embedding_dim": 32, "epochs": len(self.history["epoch"])},
            "transformer",
        )

        return metrics

    def evaluate_validation(self, data, batch_size=128):
        if self.model is None:
            raise Exception(
                "Transformer model cannot be evaluated without being trained"
            )

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        val_loader = list(get_dataloader(data, batch_size, self.tokenizer))
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                logits = self.model(input_ids)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total,
        }
