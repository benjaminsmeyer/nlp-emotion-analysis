import re
from collections import Counter
import torch
from datasets import load_dataset

LABEL_NAMES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

RANDOM_SEED = 42

_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9 ]")
_RE_MULTI_SPACE = re.compile(r" +")


def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = _RE_NON_ALPHANUM.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text


def load_emotion_data():
    """Load dair-ai/emotion dataset and print split/class statistics.

    Returns:
        (train_dataset, val_dataset, test_dataset) as HuggingFace Dataset objects
    """
    dataset = load_dataset("dair-ai/emotion")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    print(f"Dataset splits — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    label_counts = Counter(train_dataset["label"])
    total = len(train_dataset)
    print("Train class distribution:")
    for label_id, name in LABEL_NAMES.items():
        count = label_counts.get(label_id, 0)
        pct = count / total * 100
        print(f"  {label_id} ({name}): {count} ({pct:.1f}%)")

    return train_dataset, val_dataset, test_dataset


def get_dataloader(split, batch_size=32, tokenizer=None):
    """Return batches from the given split.

    Args:
        split: HuggingFace Dataset split
        batch_size: number of examples per batch
        tokenizer: if None, yields (texts, labels) tuples of raw strings;
                   if provided, yields tokenized tensor dicts with 'labels'
    """
    texts = split["text"]
    labels = split["label"]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        if tokenizer is None:
            yield batch_texts, batch_labels
        else:
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding["labels"] = torch.tensor(batch_labels)
            yield encoding
