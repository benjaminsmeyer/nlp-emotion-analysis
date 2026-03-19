import argparse
import importlib
import random

import numpy as np
import torch
from tabulate import tabulate
from data.data_loader import load_emotion_data, LABEL_NAMES
from results import exporter

DL_MODELS = {"rnn", "transformer"}

MODEL_REGISTRY = {
    "naive_bayes": ("models.naive_bayes", "NaiveBayesModel"),
    "logistic_regression": ("models.logistic_regression", "LogisticRegressionModel"),
    "rnn": ("models.rnn", "RNNModel"),
    "transformer": ("models.transformer", "TransformerModel"),
}

SAMPLE_TEXTS = [
    "i feel so happy today",
    "this makes me really angry",
    "i am scared of the dark",
    "i love spending time with you",
    "i feel so sad and alone",
    "wow that was totally unexpected",
]


def load_model(model_key: str):
    module_path, class_name = MODEL_REGISTRY[model_key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def run_model(model_key, train_data, val_data, test_data):
    print(f"Running: {model_key}")

    try:
        model = load_model(model_key)
    except (ImportError, ModuleNotFoundError):
        print(f"  [SKIP] {model_key} model not found or failed to import — implement models/{model_key}.py to enable")
        return None
    except Exception as e:
        print(f"  [ERROR] Failed to load {model_key}: {e}")
        return None

    model.time_training(train_data, val_data)
    print(f"  Training time: {model.training_time_seconds:.2f}s")

    metrics = model.evaluate(test_data)
    print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

    model.time_inference(SAMPLE_TEXTS)
    print(f"  Inference time per sample: {model.inference_time_per_sample*1000:.3f}ms")

    y_true = test_data["label"]
    y_pred = model.predict(test_data["text"])
    exporter.save_confusion_matrix(y_true, y_pred, model_key, LABEL_NAMES, f1_score=metrics['f1'])

    if model_key in DL_MODELS and hasattr(model, "history"):
        exporter.save_training_curves(model.history, model_key)

    return {
        **metrics,
        "training_time_seconds": model.training_time_seconds,
        "inference_time_per_sample": model.inference_time_per_sample,
    }


def print_comparison_table(results: dict) -> None:
    try:
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Train Time (s)", "Infer (ms/sample)"]
        rows = []
        for model_name, m in results.items():
            rows.append([
                model_name,
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['training_time_seconds']:.2f}",
                f"{m['inference_time_per_sample']*1000:.3f}",
            ])
        print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        print("\nModel Results:")
        for model_name, m in results.items():
            print(
                f"  {model_name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                f"train={m['training_time_seconds']:.2f}s "
                f"infer={m['inference_time_per_sample']*1000:.3f}ms"
            )


def main():
    parser = argparse.ArgumentParser(description="NLP Emotion Analysis — CS4120")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model(s) to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset...")
    train_data, val_data, test_data = load_emotion_data()

    results = {}
    for model_key in models_to_run:
        result = run_model(model_key, train_data, val_data, test_data)
        if result is not None:
            results[model_key] = result

    if results:
        completed_models = list(results.keys())
        print_comparison_table(results)
        exporter.save_runtime_comparison(results, completed_models)
        exporter.save_metrics_comparison(results, completed_models)
        if set(results.keys()) == set(MODEL_REGISTRY.keys()):
            exporter.save_all_models_comparison(results)
    else:
        print("\nNo models completed successfully.")


if __name__ == "__main__":
    main()
