import torch.nn as nn

class NeuralNetMLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
        # First hidden layer
        nn.Linear(in_features=784, out_features=128),
        nn.ReLU(),

        # Second hidden layer
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),

        # Output layer (without Softmax Layer due to CrossEntropyLoss conflict)
        nn.Linear(in_features=64, out_features=10)
    )

  def forward(self, X):
    return self.network(X)