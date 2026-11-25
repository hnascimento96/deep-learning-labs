import torch

BATCH_SIZE = 32
LR = 0.1
EPOCHS = 20

OPTIM = "SGD"

OPTIMIZER_DICT = {
    "SGD" : torch.optim.SGD
}