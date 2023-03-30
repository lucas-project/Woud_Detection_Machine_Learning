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
import random

with open('C:/Users/User/Desktop/ml/export-2023-03-23T01_28_26.157Z.json', 'r') as file:
    labelbox_data = json.load(file)




def load_images_filepaths(image_path, mask_path):
    filepaths = []

    for file in os.listdir(image_path):
        image_file = os.path.join(image_path, file)
        mask_file = os.path.join(mask_path, os.path.splitext(file)[0] + "_mask.png")
        
        if not os.path.exists(mask_file):
            print(f"Mask file not found: {mask_file}")
            continue
        
        filepaths.append((image_file, mask_file))

    return filepaths

image_size = (256, 256)
image_path = "C:\\Users\\User\\Desktop\\ml\\dataset"
mask_path = "C:\\Users\\User\\Desktop\\ml\\masking"

# Call the updated function to get the filepaths
filepaths = load_images_filepaths(image_path, mask_path)

# Shuffle the filepaths
random.shuffle(filepaths)

# Split ratio for validation set (20% validation)
validation_split = 0.2
split_index = int(len(filepaths) * validation_split)

# Split the data into training and validation sets
validation_data = filepaths[:split_index]
training_data = filepaths[split_index:]

# Print the length of the training and validation sets
print("Training set size:", len(training_data))
print("Validation set size:", len(validation_data))

# Now, modify the load_images function to accept the list of filepaths and load the corresponding images and masks
def load_images(filepaths, image_size):
    images = []
    masks = []

    for image_file, mask_file in filepaths:
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Failed to read image file: {image_file}")
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

# Load the training and validation images and masks
X_train, y_train = load_images(training_data, image_size)
X_test, y_test = load_images(validation_data, image_size)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)




#images, masks = load_images(image_path, mask_path, image_size)

# Split the dataset into training and testing sets:
# X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
# X_train, y_train = images, masks
# X_test, y_test = images, masks
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


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

history = model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, reduce_lr])


# Visualize the training progress:
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Visualize the training progress:


# plt.figure(figsize=(12, 5))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 5))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


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
