import time
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaseEmotionModel(ABC):
    def __init__(self):
        self.training_time_seconds = None
        self.inference_time_per_sample = None

    # Abstract interface — every model must implement these

    @abstractmethod
    def train(self, train_data, val_data) -> None:
        """Train the model on train_data, using val_data for validation."""

    @abstractmethod
    def evaluate(self, test_data) -> dict:
        """Evaluate on test_data; return metrics dict from get_metrics()."""

    @abstractmethod
    def predict(self, texts: list[str]) -> list[int]:
        """Return predicted label indices for a list of raw text strings."""

    # Shared concrete methods

    def time_training(self, train_data, val_data) -> None:
        """Run train() and record wall-clock time in self.training_time_seconds."""
        start = time.perf_counter()
        self.train(train_data, val_data)
        self.training_time_seconds = time.perf_counter() - start

    def time_inference(self, texts: list[str]) -> list[int]:
        """Run predict() and record per-sample wall-clock time."""
        start = time.perf_counter()
        predictions = self.predict(texts)
        elapsed = time.perf_counter() - start
        self.inference_time_per_sample = elapsed / len(texts) if texts else 0.0
        return predictions

    def get_metrics(self, y_true, y_pred) -> dict:
        """Compute accuracy, macro precision, recall, and F1."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def get_model_name(self) -> str:
        return self.__class__.__name__
