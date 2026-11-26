import torch
import torch.nn as nn
import torchvision
import numpy as np
from src.config import OPTIMIZER_DICT
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def get_dataset(train=False):
    return torchvision.datasets.MNIST(
        root='./data',
        train=train,
        download=True
)

def preprocess(raw_dataset):

    data_input = np.array([np.array(matrix).flatten() for matrix, _ in raw_dataset])/255

    labels = [label for _, label in raw_dataset]

    # To torch.tensor
    test_input = torch.tensor(data_input, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    # To dataset

    tensor_dataset = TensorDataset(test_input, labels)

    return tensor_dataset

def train(model, batches, lr, optim_name, epochs, validation_dataset):
    optimizer = OPTIMIZER_DICT[optim_name](model.parameters(), lr=lr)

    # Initializing loss criterion
    criterion = nn.CrossEntropyLoss()

    mean_loss = []
    loss_val = []
    print("-"*100)
    print("Training neural network")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0
        loss_val_epoch = 0
        for X_train, y_train in batches:
            # Resets gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(X_train)

            # Computing loss
            loss = criterion(output, y_train)

            # Backward
            loss.backward()

            # Updates params
            optimizer.step()

            total_loss_epoch += loss.item()

        mean_loss.append(total_loss_epoch/len(batches))

        model.eval()
        with torch.no_grad():
            X_val, y_val = validation_dataset[:]
            output = model(X_val)
            loss_val.append(criterion(output, y_val).item())

        if ((epoch == 1) or ((epoch % 5) == 0)):
            print(f"Epoch {epoch}. Mean loss: {total_loss_epoch/len(batches):.4f}")

    return mean_loss, loss_val

def evaluation(model, dataset):
    model.eval()

    softmax_layer = nn.Softmax(dim=-1)

    no_correct = 0
    for X_train, y_train in dataset:
        with torch.no_grad():
            output_logit = model(X_train)
            output = softmax_layer(output_logit)

            if (torch.argmax(output) == y_train):
                no_correct += 1

    accuracy = no_correct/len(dataset)*100.00

    return accuracy




