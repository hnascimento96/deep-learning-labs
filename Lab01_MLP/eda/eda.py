import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset_train = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True
)

print(f"Length of dataset_train: {len(dataset_train)}")
print(f"dataset_train[0]: {dataset_train[0]}")


image, label = dataset_train[0]

plt.imshow(image)
plt.title(f"Label: {label}")
plt.show()

fig, axes = plt.subplots(nrows=4, ncols=3)

for row in range(4):
  for col in range(3):
    image, label = dataset_train[row*3+col]

    axes[row,col].imshow(image)
    axes[row,col].set_title(f"Label: {label}")

plt.tight_layout()
plt.show()

labels = np.array([data[1] for data in dataset_train])

y_train = pd.DataFrame(labels)

y_train.describe()

y_train.value_counts(normalize=True, ascending=True)

y_train.hist(bins=10)
plt.title('Histogram')
plt.show()