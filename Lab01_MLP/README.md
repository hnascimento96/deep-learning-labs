# Lab 01 — Multilayer Perceptron on MNIST

Handwritten digit classification (0–9) with a fully connected neural network in PyTorch.
Companion lab to the first lecture of **MIT 6.S191 – Introduction to Deep Learning**
(perceptron, activation functions, loss, gradient descent, backpropagation, mini-batches, overfitting).

## Goal

Study how the **training dynamics** of an MLP respond to its hyperparameters
(learning rate, optimizer, batch size, training set size) by comparing training and
validation loss curves. The focus is understanding the behavior, not finding the best model.

## Model

| Layer | Shape | Activation |
|---|---|---|
| Input | 784 (28 × 28 pixels, flattened, scaled to [0, 1]) | — |
| Hidden 1 | 784 → 128 | ReLU |
| Hidden 2 | 128 → 64 | ReLU |
| Output | 64 → 10 (logits) | — |

The output layer has no softmax: `nn.CrossEntropyLoss` applies log-softmax internally.

## Data split

| Set | Source | Size | Used for |
|---|---|---|---|
| Train | First `size_train` samples of the MNIST training set | up to 50,000 | Weight updates |
| Validation | Last 10,000 samples of the MNIST training set | 10,000 | Loss curve per epoch |
| Test | MNIST test set | 10,000 | Final accuracy only |

> `size_train` must be ≤ 50,000 so that the training and validation sets do not overlap.

The validation loss is computed **once per epoch**, after all weight updates of that epoch,
with `model.eval()` and `torch.no_grad()` (forward pass only, no weight updates).

## Project structure

```
Lab01_MLP/
├── eda/
│   └── eda.py               # Exploratory analysis: samples and class distribution
├── src/
│   ├── config.py            # Default hyperparameters and optimizer registry
│   ├── NeuralNetMLP.py      # Model definition
│   ├── utils.py             # Data loading, preprocessing, training and evaluation
│   └── mnsti_classifier.py  # run_pipeline(): data → training → evaluation
└── experiment.ipynb         # Runs and compares experiments
```

## How to run

Install the dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib jupyter
```

Run everything **from the `Lab01_MLP/` folder**, so that `from src...` imports and the
`./data` path resolve correctly. MNIST is downloaded automatically to `./data` on the first run.

```bash
cd Lab01_MLP
jupyter notebook experiment.ipynb
```

Each experiment is a single call:

```python
from src.mnsti_classifier import run_pipeline

train_loss, val_loss, train_accuracy, test_accuracy = run_pipeline(
    batch_size=32, lr=0.01, optim_name="SGD", size_train=50000, epochs=20
)
```

| Argument | Description |
|---|---|
| `batch_size` | Samples per weight update |
| `lr` | Learning rate |
| `optim_name` | Key in `OPTIMIZER_DICT` (`src/config.py`) |
| `size_train` | Number of training samples (≤ 50,000) |
| `epochs` | Passes over the training set |

Returns the per-epoch training and validation losses, plus the final train and test accuracy (%).
The random seed is fixed, so runs with the same arguments are reproducible.

## Experiments

One hyperparameter is varied at a time; everything else stays at the baseline.

| Experiment | Values | Question |
|---|---|---|
| Learning rate | 0.001, 0.01, 0.1, 1.0 | Why is there a "sweet spot" for the step size? |
| Optimizer | SGD, SGD + momentum, Adam (each at its typical LR) | What do momentum and adaptive steps fix? |
| Batch size | 8, 32, 128, 1024 | What is gained and lost by estimating the gradient from fewer samples? |
| Training set size | 500, 2,000, 10,000, 50,000 | When does the model memorize instead of learning? |

## Results

### Baseline sanity run

SGD, LR 0.01, batch 32, 50,000 training samples, 20 epochs.

| Train accuracy | Test accuracy |
|---|---|
| 97.68% | 96.68% |

The validation loss starts *below* the training loss in the first epoch. This is expected:
the training loss is averaged while the weights are still being updated, whereas the
validation loss is measured on the model at the end of the epoch.

### Hyperparameter experiments

_To be added._

## Conclusions

_To be added._
