import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image, ImageDraw
from torchvision.models import resnet18

# Create directories if they don't exist
os.makedirs("data/train/shape1", exist_ok=True)
os.makedirs("data/train/shape2", exist_ok=True)
os.makedirs("data/val/shape1", exist_ok=True)
os.makedirs("data/val/shape2", exist_ok=True)

def create_circle_image(file_path, color):
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((64, 64, 192, 192), fill=color)
    img.save(file_path)

def create_rectangle_image(file_path, color):
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((64, 64, 192, 192), fill=color)
    img.save(file_path)

# Create circle images
for i in range(10):
    create_circle_image(f"data/train/shape1/circle_{i}.jpg", (255, 0, 0))
    create_circle_image(f"data/val/shape1/circle_{i}.jpg", (255, 0, 0))

# Create rectangle images
for i in range(10):
    create_rectangle_image(f"data/train/shape2/rectangle_{i}.jpg", (0, 255, 0))
    create_rectangle_image(f"data/val/shape2/rectangle_{i}.jpg", (0, 255, 0))

# Hyperparameters
num_epochs = 25
batch_size = 16
learning_rate = 0.001

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root='data/train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root='data/val', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Define the model
model = resnet18(pretrained=True)

# Modify the last layer for the number of classes in your dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'model.ckpt')

# Load the saved model
loaded_model = resnet18(pretrained=False)
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)
loaded_model.load_state_dict(torch.load('model.ckpt'))

# Move the loaded model to the device (GPU or CPU)
loaded_model.to(device)

# Set the loaded model to evaluation mode
loaded_model.eval()
