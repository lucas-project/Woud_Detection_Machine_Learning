# 1. Load and preprocess the data
# First, we'll load the images from the dataset and preprocess them using image segmentation 
# techniques to identify the wound border. We'll use the OpenCV library for image processing.

import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class WoundDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        image = cv2.imread(image_path)

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to segment wound area
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours of wound area
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None
        c = max(contours, key=cv2.contourArea)

        # Draw wound area and extract it from the original image
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], 0, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Resize image and mask to fixed size
        resized = cv2.resize(masked, (256, 256))
        mask_resized = cv2.resize(mask, (256, 256))

        # Convert image and mask to PyTorch tensors
        image_tensor = torch.from_numpy(resized.transpose((2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float() / 255.0

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor

image_folder = "C:\\Users\\User\\Desktop\\dataset"
dataset = WoundDataset(image_folder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Define the model architecture
# Next, we'll define the model architecture for image segmentation. We'll use a U-Net architecture, 
# which is a popular architecture for image segmentation that consists of an encoder (which 
# downsamples the image) and a decoder (which upsamples the image).

import torch.nn as nn
import torch.nn.functional as F

class WoundSegmentationNet(nn.Module):
    def __init__(self):
        super(WoundSegmentationNet, self).__init__()

        # Encoder
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout2d(p=0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)  # Added convolutional layer
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)


        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.conv2_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_4 = nn.BatchNorm2d(128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1_3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.bn1_1(self.conv1_1(x)))
        conv1 = F.relu(self.bn1_2(self.conv1_2(conv1)))
        pool1 = self.pool1(conv1)

        conv2 = F.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2 = F.relu(self.bn2_2(self.conv2_2(conv2)))
        pool2 = self.pool2(conv2)

        conv3 = F.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3 = F.relu(self.bn3_2(self.conv3_2(conv3)))
        pool3 = self.pool3(conv3)

        conv4 = F.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4 = F.relu(self.bn4_2(self.conv4_2(conv4)))
        drop4 = self.dropout(conv4)
        pool4 = self.pool4(drop4)

        # Decoder
        upconv3 = self.upconv3(pool4)
        merge3 = torch.cat([upconv3, conv3], dim=1)
        conv3 = F.relu(self.bn3_3(self.conv3_3(merge3)))
        conv3 = F.relu(self.bn3_4(self.conv3_4(conv3)))

        upconv2 = self.upconv2(conv3)
        merge2 = torch.cat([upconv2, conv2], dim=1)
        conv2 = F.relu(self.bn2_3(self.conv2_3(merge2)))
        conv2 = F.relu(self.bn2_4(self.conv2_4(conv2)))

        upconv1 = self.upconv1(conv2)
        merge1 = torch.cat([upconv1, conv1], dim=1)
        conv1 = F.relu(self.bn1_3(self.conv1_3(merge1)))
        conv1 = F.relu(self.bn1_4(self.conv1_4(conv1)))

        output = self.conv1_5(conv1)
        output = torch.sigmoid(output)

        return output



# 3. Define the classification model
# Once we've identified the wound border using image segmentation, we can use a separate 
# classification model to classify the wound as either healed or non-healed based on the 
# difference in the wound border over time. We'll define a simple model using PyTorch that 
# consists of two fully connected layers:

class WoundClassificationNet(nn.Module):
    def __init__(self):
        super(WoundClassificationNet, self).__init__()

        self.fc1 = nn.Linear(256 * 256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(-1, 256 * 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. Train the model
# Now that we've defined our segmentation and classification models, we can train them on the wound
#  dataset using PyTorch.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmentation_model = WoundSegmentationNet().to(device)
classification_model = WoundClassificationNet().to(device)

criterion = nn.CrossEntropyLoss()
segmentation_optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=0.0001)
classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)

# Define the number of epochs and initialize the step counter
num_epochs = 10
i = 0

# 4. Train the model
# Now that we've defined our segmentation and classification models, we can train them on the wound
#  dataset using PyTorch.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmentation_model = WoundSegmentationNet().to(device)
classification_model = WoundClassificationNet().to(device)

criterion = nn.CrossEntropyLoss()
segmentation_optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=0.0001)
classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)

# Define the number of epochs and initialize the step counter
num_epochs = 10
i = 0

# Train the models
for epoch in range(num_epochs):
    for images, masks in dataloader:
        # Move the input images and masks to the device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass through segmentation model and compute segmentation loss
        segmentation_optimizer.zero_grad()
        outputs = segmentation_model(images)
        loss_segmentation = criterion(outputs * masks, masks)
        loss_segmentation.backward()
        segmentation_optimizer.step()

        # Forward pass through classification model and compute classification loss
        segmentation_model.eval()
        with torch.no_grad():
            masks = masks.cpu()
            outputs = outputs.detach().cpu()
            masked_images = images * masks.repeat(1, 3, 1, 1)
            embeddings = segmentation_model(masked_images)
            embeddings = embeddings.view(embeddings.size(0), -1)
            labels = torch.tensor([1 if 'non_healed' in filename else 0 for filename in dataset.image_files]).long()
            labels = labels.to(device)
            classification_optimizer.zero_grad()
            outputs = classification_model(embeddings)
            loss_classification = criterion(outputs, labels)
            loss_classification.backward()
            classification_optimizer.step()

        segmentation_model.train()

        # Print loss and accuracy
        if i % 10 == 0:
            print("Epoch {}/{}, Step {}/{}, Segmentation Loss: {:.4f}, Classification Loss: {:.4f}".format(
                epoch + 1, num_epochs, i, len(dataloader),
                loss_segmentation.item(), loss_classification.item()))

        i += 1

    # Save the models
    torch.save(segmentation_model.state_dict(), "segmentation_model.pt")
    torch.save(classification_model.state_dict(), "classification_model.pt")


# In this code, we first move the input images and masks to the device (either CPU or GPU) that 
# we're using for computation. Then we perform a forward pass through the segmentation model and
#  compute the segmentation loss. Next, we switch the segmentation model to evaluation mode and 
# use it to extract embeddings from the masked images. We then compute the classification loss using
#  the embeddings and ground truth labels, and backpropagate the loss through the classification 
# model. Finally, we switch the segmentation model back to training mode and print the losses and
#  accuracy at regular intervals.
