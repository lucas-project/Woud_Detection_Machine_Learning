# First, we'll start by loading and preprocessing the images. We'll use the same load_images_from_folder and preprocess_images functions as before.
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images):
    data = np.array([np.array(img) for img in images])
    data = data.astype('float32') / 255.
    return data

folder1 = "path/to/first/folder"
folder2 = "path/to/second/folder"

images1 = load_images_from_folder(folder1)
images2 = load_images_from_folder(folder2)

x1 = preprocess_images(images1)
x2 = preprocess_images(images2)

# Next, we'll create the labels and split the data into training and testing sets.
y1 = np.zeros(x1.shape[0])
y2 = np.ones(x2.shape[0])
y = np.concatenate((y1, y2))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((x1, x2)), y, test_size=0.2, random_state=42)

# We'll create a PyTorch dataset and dataloader for the training and testing sets.

class WoundDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

img_width, img_height = 224, 224

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_width, img_height)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
])

train_dataset = WoundDataset(x_train, y_train, transform=train_transforms)
test_dataset = WoundDataset(x_test, y_test, transform=test_transforms)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# We'll define the model using PyTorch's nn module.
import torch.nn as nn
import torch.nn.functional as F

class WoundNet(nn.Module):
    def __init__(self):
        super(WoundNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * 26 * 26, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(-1, 128 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

model = WoundNet()

# We'll use binary cross-entropy loss and Adam optimizer.
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# We'll train the model for 50 epochs.
num_epochs = 50

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_acc += (torch.round(outputs.squeeze()) == labels).sum().item()

    train_loss /= len(train_dataloader.dataset)
    train_acc /= len(train_dataloader.dataset)

    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader):
            outputs = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            test_loss += loss.item() * images.size(0)
            test_acc += (torch.round(outputs.squeeze()) == labels).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_acc /= len(test_dataloader.dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Finally, we can use the model to predict the difference between a patient's wound at different times.

def predict_image(img_path):
    img = Image.open(img_path)
    img = img.resize((img_width, img_height))
    img = np.array(img)
    img = test_transforms(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    if output < 0.5:
        print("The wound has not healed.")
    else:
        print("The wound has healed.")

img_path = "path/to/test/image"
predict_image(img_path)


