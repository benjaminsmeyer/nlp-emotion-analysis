# NLP Emotion Analysis

**CS4120 — Natural Language Processing**

**Team members:** Benjamin Meyer, Nico Rising, Ami Ashman, Bradley Harvan

## Research Question

Can we accurately classify emotions expressed in short social media text using a range of approaches from classical machine learning to modern deep learning, and how do these approaches compare in accuracy and efficiency?

## Dataset

**dair-ai/emotion** — ~20,000 English tweets labeled with 6 emotions:

| Label | Emotion  | ~Train % |
| ----- | -------- | -------- |
| 0     | sadness  | 28.5%    |
| 1     | joy      | 32.5%    |
| 2     | love     | 8.0%     |
| 3     | anger    | 12.5%    |
| 4     | fear     | 12.5%    |
| 5     | surprise | 6.0%     |

Splits: 16,000 train / 2,000 validation / 2,000 test.

## Models

| Model               | File                            | Team Member    | Description                                   |
| ------------------- | ------------------------------- | -------------- | --------------------------------------------- |
| Naive Bayes         | `models/naive_bayes.py`         | Benjamin Meyer | Multinomial NB on TF-IDF features             |
| Logistic Regression | `models/logistic_regression.py` | Ami Ashman     | L2-regularized LR on TF-IDF features          |
| RNN                 | `models/rnn.py`                 | Bradley Harvan | Bidirectional LSTM with pretrained embeddings |
| Transformer         | `models/transformer.py`         | Nico Rising    | Fine-tuned DistilBERT (HuggingFace)           |

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# Run all models
python main.py --model all

# Run a single model
python main.py --model naive_bayes
python main.py --model logistic_regression
python main.py --model rnn
python main.py --model transformer

# Custom seed
python main.py --model all --seed 123
```

## Output Files

All results are saved to `results/`:

| File                                   | Description                                       |
| -------------------------------------- | ------------------------------------------------- |
| `results/{model}/confusion_matrix.png` | Per-class prediction breakdown                    |
| `results/{model}/training_curves.png`  | Loss and accuracy curves (DL models only)         |
| `results/metrics_comparison.png`       | Grouped bar chart of accuracy/precision/recall/F1 |
| `results/runtime_comparison.png`       | Training time and inference latency per model     |

## Evaluation Metrics

- **Accuracy** — fraction of all predictions that are correct
- **Macro Precision** — average per-class precision, treating each class equally regardless of support
- **Macro Recall** — average per-class recall, measuring how well each emotion is detected
- **Macro F1** — harmonic mean of precision and recall, the primary comparison metric

## Implementing a New Model

Create `models/your_model.py` inheriting from `BaseEmotionModel`:

```python
from models.base_model import BaseEmotionModel

class YourModel(BaseEmotionModel):
    def train(self, train_data, val_data): ...
    def evaluate(self, test_data): ...
    def predict(self, texts: list[str]) -> list[int]: ...
```

Then add it to `MODEL_REGISTRY` in `main.py`.
