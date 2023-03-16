import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def visualize_dataset(training_data):
    labels_map = {
        0: 'algal leaf',
        1: 'Anthracnose',
        2: 'bird eye spot',
        3: 'brown blight',
        4: 'gray light',
        5: 'healthy',
        6: 'red leaf spot',
        7: 'white spot'
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()

# create dataset
dataset = datasets.ImageFolder(
    root="tea_dataset_merged",
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]))

# split into train/test/validation
num_samples = len(dataset)  # total number of examples
val_samples = int(0.2 * num_samples)
test_samples = int(0.1 * num_samples)
indices = np.arange(num_samples)
np.random.shuffle(indices)
val_set = torch.utils.data.Subset(dataset, indices[:val_samples])
test_set = torch.utils.data.Subset(dataset, indices[val_samples:test_samples+val_samples])
train_set = torch.utils.data.Subset(dataset, indices[test_samples+val_samples:])

print("train", len(train_set), "val", len(val_set), "test", len(test_set))

batch_size = 32

# dataset loader
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=batch_size,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=batch_size,
                                          shuffle=True)

#visualize_dataset(train_set)
#visualize_dataset(val_set)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 100

for epoch in range(epochs):
    # train
    running_loss = 0.0
    running_acc = 0.0
    counter = 0
    model.train()
    for i, data in enumerate(train_loader):
        counter += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, max_class = outputs.max(dim=1)
        running_acc += (max_class == labels).sum().item() / inputs.size(0)

    epoch_loss = running_loss / counter
    epoch_acc = running_acc / counter
    print(f'[{epoch}/{epochs}, {inputs.shape[0]}], train loss: {epoch_loss}, train accuracy: {epoch_acc}')

    # validation
    running_loss = 0.0
    running_acc = 0.0
    counter = 0
    model.eval()
    for i, data in enumerate(val_loader):
        counter += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        running_loss += loss.item()
        _, max_class = outputs.max(dim=1)
        running_acc += (max_class == labels).sum().item() / inputs.size(0)

    epoch_loss = running_loss / counter
    epoch_acc = running_acc / counter
    print(f'[{epoch}/{epochs}, {inputs.shape[0]}], validation loss: {epoch_loss}, validation accuracy: {epoch_acc}')

print('Finished Training')
