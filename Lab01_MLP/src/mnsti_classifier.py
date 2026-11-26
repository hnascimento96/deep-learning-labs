
import torch
from torch.utils.data import DataLoader, Subset
from src.utils import get_dataset, train, evaluation, preprocess
from src.NeuralNetMLP import NeuralNetMLP

def run_pipeline(batch_size, lr, optim_name, size_train, epochs):

    torch.manual_seed(seed=56)

    dataset_train = get_dataset(train=True)

    dataset_train = preprocess(dataset_train)

    dataset_validation = Subset(dataset_train, range(len(dataset_train) - 10000, len(dataset_train)))
    dataset_train = Subset(dataset_train, range(0,size_train))

    train_batches = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    model = NeuralNetMLP()

    ########## TRAINING ###############

    train_loss, val_loss = train(model=model, lr=lr, optim_name=optim_name, epochs=epochs,
                                 batches=train_batches,
                                 validation_dataset=dataset_validation)

    ############# EVALUATION ##########

    # Train evaluation

    train_accuracy = evaluation(model, dataset_train)
    print(f"Train accuracy: {train_accuracy:.2f}%")

    # Test evaluation

    test_dataset = get_dataset(train=False)

    test_dataset = preprocess(test_dataset)

    test_accuracy = evaluation(model, test_dataset)

    print(f"Test accuracy: {test_accuracy:.2f}%")

    return train_loss, val_loss, train_accuracy, test_accuracy