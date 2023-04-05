import os
import json
import numpy as np
import cv2
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
import re


# Set up directories
images_json_path = 'C:/Users/User/Desktop/ml/json_images/'
masks_json_path = 'C:/Users/User/Desktop/ml/json_masking/'
images_png_path = 'C:/Users/User/Desktop/ml/png_images/'
masks_png_path = 'C:/Users/User/Desktop/ml/png_masking/'
evaluation_path = 'C:/Users/User/Desktop/ml/evaluation/'

# Load and preprocess the data
def load_images_and_masks(images_json_path, images_png_path, masks_json_path, masks_png_path):
    images = []
    masks = []

    # Load images with JSON masks
    json_files = os.listdir(masks_json_path)
    
    # Sort JSON files numerically
    json_files = sorted(json_files, key=lambda x: int(re.search(r'\d+', x).group()))

    for file in json_files:
        # Load corresponding JSON mask
        with open(os.path.join(masks_json_path, file[:-5] + '.json')) as f:
            mask_json = json.load(f)

        if 'Label' in mask_json:
            # Load image
            image = cv2.imread(os.path.join(images_json_path, file[:-5] + '.jpg'), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            
            # Add additional channel
            mean_pixel_value = np.mean(image, axis=-1, keepdims=True)
            image = np.concatenate([image, mean_pixel_value], axis=-1)
            
            images.append(image)

            polygon = mask_json['Label']['objects'][0]['instanceURI']
            response = requests.get(polygon)
            mask = np.array(Image.open(BytesIO(response.content)).convert('L'))
            mask = cv2.resize(mask, (256, 256))
            mask = np.expand_dims(mask, axis=-1)  # Convert to (256, 256, 1) shape
            mask = mask / 255.0  # Scale mask values between 0 and 1
        else:
            key = file + str(os.stat(os.path.join(images_json_path, file)).st_size)
            regions = mask_json['_via_img_metadata'][key]['regions']
            
            if len(regions) > 0:
                # Load image
                image = cv2.imread(os.path.join(images_json_path, file[:-5] + '.jpg'), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256))
                
                # Add additional channel
                mean_pixel_value = np.mean(image, axis=-1, keepdims=True)
                image = np.concatenate([image, mean_pixel_value], axis=-1)
                
                images.append(image)

                points = regions[0]['shape_attributes']['all_points_x_y']
                mask = np.zeros((256, 256), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(points, np.int32).reshape((-1, 1, 2))], 255)
                mask = np.expand_dims(mask, axis=-1)  # Convert to (256, 256, 1) shape
            else:
                print(f"No regions found for {file}. Skipping...")
                continue

        masks.append(mask)
        print(f"Loaded JSON mask for file {file}")

    # Load images with PNG masks
    png_files = os.listdir(images_png_path)

    # Sort PNG files numerically
    png_files = sorted(png_files, key=lambda x: int(re.search(r'\d+', x).group()))

    for file in png_files:
        # Load PNG mask
        mask = cv2.imread(os.path.join(masks_png_path, file[:-4] + '.png'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256,256))
        mask = np.expand_dims(mask, axis=-1)  # Convert to (256, 256, 1) shape
        mask = mask / 255.0  # Scale mask values between 0 and 1

        # Load image
        image = cv2.imread(os.path.join(images_png_path, file), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        # Add additional channel
        mean_pixel_value = np.mean(image, axis=-1, keepdims=True)
        image = np.concatenate([image, mean_pixel_value], axis=-1)
        
        images.append(image)
        masks.append(mask)
        print(f"Loaded PNG mask for file {file}")

    return np.array(images, dtype=object), np.array(masks, dtype=object)


X, y = load_images_and_masks(images_json_path, images_png_path, masks_json_path, masks_png_path)


# Display JSON masking images
def display_json_masks(masks_json_path, masks):
    for idx, file in enumerate(os.listdir(masks_json_path)):
        with open(os.path.join(masks_json_path, file)) as f:
            mask_json = json.load(f)

        if 'Label' in mask_json and 'objects' in mask_json['Label']:
            instance_uri = mask_json['Label']['objects'][0]['instanceURI']
            response = requests.get(instance_uri)
            mask = np.array(Image.open(BytesIO(response.content)).convert('L'))
            mask = cv2.resize(mask, (256, 256))
        else:
            print(f"No 'Label' or 'objects' key found for {file}. Skipping...")
            continue

        cv2.imshow(f"JSON Mask {idx}", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Display the JSON format masking images
# display_json_masks(masks_json_path, y)


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
data_gen_args = dict(rotation_range=20,  # Increase rotation range
                     width_shift_range=0.1,  # Increase width shift range
                     height_shift_range=0.1,  # Increase height shift range
                     shear_range=0.1,  # Increase shear range
                     zoom_range=0.1,  # Increase zoom range
                     brightness_range=(0.9, 1.1),
                     horizontal_flip=True,
                     vertical_flip=True,  # Add vertical flipping
                     fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_datagen.fit(X_train, augment=True, seed=42)
mask_datagen.fit(y_train, augment=True, seed=42)

# Build the model (U-Net)


def build_unet(input_shape=(256, 256, 4)):
    inputs = tf.keras.Input(input_shape)

    conv1 = Conv2D(26, (3, 3), activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(51, (3, 3), activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(102, (3, 3), activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(205, (3, 3), activation='relu', padding='same')(pool3)
    bn4 = BatchNormalization()(conv4)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(bn4), conv3])
    conv7 = Conv2D(102, (3, 3), activation='relu', padding='same')(up7)
    bn7 = BatchNormalization()(conv7)

    up8 = Concatenate()([UpSampling2D(size=(2, 2))(bn7), conv2])
    conv8 = Conv2D(51, (3, 3), activation='relu', padding='same')(up8)
    bn8 = BatchNormalization()(conv8)

    up9 = Concatenate()([UpSampling2D(size=(2, 2))(bn8), conv1])
    conv9 = Conv2D(26, (3, 3), activation='relu', padding='same')(up9)
    bn9 = BatchNormalization()(conv9)

    output = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return tf.keras.Model(inputs=inputs, outputs=output)





model = build_unet()
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


# Compile the model
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[dice_coefficient])

# Train the model
batch_size = 8  # Set your desired batch size

train_generator = zip(image_datagen.flow(X_train, batch_size=batch_size, seed=42),
                      mask_datagen.flow(y_train, batch_size=batch_size, seed=42))

val_generator = zip(image_datagen.flow(X_val, batch_size=batch_size, seed=42),
                    mask_datagen.flow(y_val, batch_size=batch_size, seed=42))

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

model.fit(train_generator, steps_per_epoch=len(X_train) // batch_size, validation_data=val_generator,
          validation_steps=len(X_val) // batch_size, epochs=50, callbacks=[early_stopping])

# Evaluate the model's performance on the evaluation set
def load_evaluation_images(evaluation_path, additional_input):
    eval_images = []
    for file in os.listdir(evaluation_path):
        image = cv2.imread(os.path.join(evaluation_path, file))
        image = cv2.resize(image, (256, 256))  # Resize the evaluation images to (256, 256)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add the additional input channel
        additional_channel = np.full((256, 256, 1), additional_input)
        image = np.concatenate([image, additional_channel], axis=-1)
        
        eval_images.append(image)
    return np.array(eval_images)


additional_input = 128  # Replace this with the desired value
evaluation_images = load_evaluation_images(evaluation_path, additional_input)


# Predict and display results
predicted_masks = model.predict(evaluation_images)
def convert_image_for_display(image):
    return np.uint8(image[:, :, :3])


for i in range(len(evaluation_images)):
    display_image = convert_image_for_display(evaluation_images[i])
    cv2.imshow(f'Original Image {i}', display_image)
    cv2.imshow(f'Predicted Mask {i}', predicted_masks[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Save the model
model.save('wound_segmentation_model.h5')


