
from torch.utils.data import TensorDataset, DataLoader
from src.config import EPOCHS, BATCH_SIZE, LR, OPTIM
from src.utils import get_dataset, train, evaluation, preprocess, plot_loss
from src.NeuralNetMLP import NeuralNetMLP

dataset_train = get_dataset(train=True)

dataset = preprocess(dataset_train)

batches = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = NeuralNetMLP()

########## TRAINING ###############

train_loss = train(model=model, lr=LR, optim_name=OPTIM, epochs=EPOCHS, batches=batches)

############# EVALUATION ###############

# Train evaluation

accuracy = evaluation(model, dataset)
print(f"Train accuracy: {accuracy:.2f}%")

# Test evaluation

test_dataset = get_dataset(train=False)

test_dataset = preprocess(test_dataset)

accuracy = evaluation(model, test_dataset)

print(f"Test accuracy: {accuracy:.2f} %")

####### Plots ##########
plot_loss(train_loss, epochs=EPOCHS, title="Train loss")