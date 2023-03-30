import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split

# Set up directories
images_path = 'C:/Users/User/Desktop/ml/images/'
masks_path = 'C:/Users/User/Desktop/ml/json/'
evaluation_path = 'C:/Users/User/Desktop/ml/evaluation/'

# Load and preprocess the data
def load_images_and_masks(images_path, masks_path):
    images = []
    masks = []

    for file in os.listdir(images_path):
        image = cv2.imread(os.path.join(images_path, file))
        image = cv2.resize(image, (64, 64))
        images.append(image)

        mask_path = os.path.join(masks_path, file.replace('.jpg', '.json'))
        with open(mask_path, 'r') as f:
            mask_json = json.load(f)
        mask = np.zeros((64, 64))
        if mask_json['_via_img_metadata'][file + str(os.path.getsize(os.path.join(images_path, file)))]['regions']:
            points = mask_json['_via_img_metadata'][file + str(os.path.getsize(os.path.join(images_path, file)))]['regions'][0]['shape_attributes']['all_points_x_y']
            cv2.fillPoly(mask, np.array([points], dtype=np.int32), 1)
        masks.append(np.expand_dims(mask, axis=-1))

    return np.array(images), np.array(masks)



X, y = load_images_and_masks(images_path, masks_path)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
data_gen_args = dict(rotation_range=10, width_shift_range=0.05, height_shift_range=0.05,
                     shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


image_datagen.fit(X_train, augment=True, seed=42)
mask_datagen.fit(y_train, augment=True, seed=42)

# Build the model (U-Net)
def build_unet(input_shape=(64, 64, 3)):
    inputs = tf.keras.Input(input_shape)

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)

    up4 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up4)

    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up5)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return tf.keras.Model(inputs=inputs, outputs=output)


model = build_unet()

# Compile the model
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[MeanIoU(num_classes=2)])

# Train the model
train_generator = zip(image_datagen.flow(X_train, batch_size=len(X_train), seed=42),
                      mask_datagen.flow(y_train, batch_size=len(y_train), seed=42))

val_generator = zip(image_datagen.flow(X_val, batch_size=len(X_val), seed=42),
                    mask_datagen.flow(y_val, batch_size=len(y_val), seed=42))

model.fit(train_generator, steps_per_epoch=len(X_train) // 2, validation_data=val_generator,
          validation_steps=len(X_val), epochs=50)

# Evaluate the model's performance on the evaluation set
def load_evaluation_images(evaluation_path):
    eval_images = []
    for file in os.listdir(evaluation_path):
        image = cv2.imread(os.path.join(evaluation_path, file))
        image = cv2.resize(image, (64, 64))  # Resize the evaluation images to (64, 64)
        eval_images.append(image)
    return np.array(eval_images)


evaluation_images = load_evaluation_images(evaluation_path)

# Predict and display results
predicted_masks = model.predict(evaluation_images)

for i in range(len(evaluation_images)):
    cv2.imshow(f'Original Image {i}', evaluation_images[i])
    cv2.imshow(f'Predicted Mask {i}', predicted_masks[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save the model
model.save('wound_segmentation_model.h5')
