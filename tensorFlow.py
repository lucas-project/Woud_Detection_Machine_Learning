import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import json
from PIL import Image
import requests
from io import BytesIO

with open('C:/Users/User/Desktop/ml/export-2023-03-22T23_32_10.505Z.json', 'r') as file:
    labelbox_data = json.load(file)

import cv2
import numpy as np
import os

def create_mask_images(labelbox_data, output_folder):
    for item in labelbox_data:
        for annotation in item['Label']['objects']:
            mask_url = annotation['instanceURI']
            response = requests.get(mask_url)
            mask = cv2.imdecode(np.asarray(bytearray(response.content), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            mask_filename = f"{os.path.splitext(item['External ID'])[0]}_mask.png"
            cv2.imwrite(os.path.join(output_folder, mask_filename), mask)

output_folder = r'C:\\Users\\User\\Desktop\\ml\\masking'
create_mask_images(labelbox_data, output_folder)

# Load and preprocess the images and masks:

import numpy as np
import os
import cv2

def load_images(image_path, mask_path, image_size):
    images = []
    masks = []

    for file in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE)
        mask_file = os.path.join(mask_path, os.path.splitext(file)[0] + "_mask.png")
        
        if not os.path.exists(mask_file):
            print(f"Mask file not found: {mask_file}")
            continue
        
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Failed to read image file: {os.path.join(image_path, file)}")
            continue
        
        if mask is None:
            print(f"Failed to read mask file: {mask_file}")
            continue

        image = cv2.resize(image, image_size)
        mask = cv2.resize(mask, image_size)

        images.append(image)
        masks.append(mask)

    images = np.array(images) / 255.0
    images = np.expand_dims(images, axis=-1)

    masks = np.array(masks) / 255.0
    masks = np.expand_dims(masks, axis=-1)

    return images, masks


image_size = (256, 256)
image_path = "C:\\Users\\User\\Desktop\\ml\\dataset"
mask_path = "C:\\Users\\User\\Desktop\\ml\\masking"

images, masks = load_images(image_path, mask_path, image_size)

# Split the dataset into training and testing sets:
# X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
X_train, y_train = images, masks
X_test, y_test = images, masks

# Define the U-Net model architecture:
def unet_model(input_size):
    inputs = Input(input_size)
    
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2D(512, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(c5))
    m6 = Concatenate(axis=3)([c4, u6])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(m6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(c6))
    m7 = Concatenate(axis=3)([c3, u7])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(m7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(c7))
    m8 = Concatenate(axis=3)([c2, u8])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(m8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(c8))
    m9 = Concatenate(axis=3)([c1, u9])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(m9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    c10 = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=c10)

    return model



input_size = (image_size[0], image_size[1], 1)
model = unet_model(input_size)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model:
batch_size = 16
epochs = 50

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode='min')

history = model.fit(X_train, y_train, validation_split=0, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, reduce_lr])


# Visualize the training progress:
# plt.figure(figsize=(12, 5))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 5))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# Visualize the training progress:
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Load the best model and evaluate on the test set:
model.load_weights('model.h5')

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")


# Make predictions and compare with ground truth:
predictions = model.predict(X_test)

# Select an index to visualize
index = 0

# Visualize the input image
plt.imshow(X_test[index].reshape(image_size), cmap='gray')
plt.title("Input Image")
plt.show()

# Visualize the ground truth mask
plt.imshow(y_test[index].reshape(image_size), cmap='gray')
plt.title("Ground Truth Mask")
plt.show()

# Visualize the predicted mask
plt.imshow(predictions[index].reshape(image_size), cmap='gray')
plt.title("Predicted Mask")
plt.show()
