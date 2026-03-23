from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tabulate import tabulate

from models.base_model import BaseEmotionModel
from data.data_loader import preprocess_text
from results import exporter


class LogisticRegressionModel(BaseEmotionModel):
    def __init__(self):
        super().__init__()
        self.vectorizer = None
        self.classifier = None
        self.best_c = None
        self.best_max_features = None

    def train(self, train_data, val_data) -> None:
        train_texts = [preprocess_text(t) for t in train_data["text"]]
        val_texts = [preprocess_text(t) for t in val_data["text"]]
        train_labels = train_data["label"]
        val_labels = val_data["label"]

        C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
        MAX_FEATURES = [5000, 10000, 20000]

        results = []
        best_f1 = -1.0
        best_vec = None

        for max_features in MAX_FEATURES:
            vec = TfidfVectorizer(max_features=max_features)
            X_train = vec.fit_transform(train_texts)
            X_val = vec.transform(val_texts)
            for c in C_VALUES:
                clf = LogisticRegression(C=c, max_iter=1000, solver="lbfgs")
                clf.fit(X_train, train_labels)
                preds = clf.predict(X_val)
                val_f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
                results.append((c, max_features, val_f1))
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    self.best_c = c
                    self.best_max_features = max_features
                    best_vec = vec

        print(tabulate(results, headers=["c", "max_features", "val_macro_f1"], floatfmt=".4f"))
        print(f"Best: c={self.best_c} max_features={self.best_max_features} → val F1={best_f1:.4f}")

        self.vectorizer = best_vec
        X_train = self.vectorizer.transform(train_texts)
        self.classifier = LogisticRegression(C=self.best_c, max_iter=1000, solver="lbfgs")
        self.classifier.fit(X_train, train_labels)

    def predict(self, texts: list[str]) -> list[int]:
        processed = [preprocess_text(t) for t in texts]
        X = self.vectorizer.transform(processed)
        return list(self.classifier.predict(X))

    def evaluate(self, test_data) -> dict:
        y_true = test_data["label"]
        y_pred = self.predict(test_data["text"])
        metrics = self.get_metrics(y_true, y_pred)

        print("Test Results — LogisticRegression:")
        print(f"  accuracy  : {metrics['accuracy']:.4f}")
        print(f"  precision : {metrics['precision']:.4f}")
        print(f"  recall    : {metrics['recall']:.4f}")
        print(f"  f1        : {metrics['f1']:.4f}")

        exporter.save_model_metrics_bar(metrics, "logistic_regression")
        exporter.save_text_summary(
            metrics,
            {"c": self.best_c, "max_features": self.best_max_features},
            "logistic_regression",
        )

        return metrics