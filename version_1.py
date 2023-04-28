import math
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
import matplotlib.pyplot as plt
from skimage import measure
from v1_border import build_unet, display_json_masks, extract_wound_area, load_images_and_masks, extract_blue_contour, process_images
from v1_coin import calculate_actual_coin_area, calculate_actual_wound_area, calculate_ratio_wound_image, display_coin_detection, detect_coin, calculate_wound_area, estimate_actual_area
from v1_colour import calculate_color_percentage, quantize_image, extract_color_information
from v1_evaluation import load_evaluation_images

# Set up path
images_json_path = 'fake_wound/'
masks_json_path = 'fake_jj/'
images_png_path = 'fake_png_1/'
masks_png_path = 'fake_png_2/'
evaluation_path = 'fake_evaluation/'
input_directory = 'contour/'  
output_directory = 'contour_processed/' 

if not os.path.exists(output_directory):
    os.makedirs(output_directory)



for image_file in os.listdir(input_directory):
    if image_file.endswith('.jpg'):
        input_path = os.path.join(input_directory, image_file)
        output_path = os.path.join(output_directory, image_file)
        contour_image, pixel_count = extract_blue_contour(input_path) 
        cv2.imwrite(output_path, contour_image)

        # Display the output image
        cv2.imshow('Contour Image', contour_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# load_images_and_masks
X, y = load_images_and_masks(images_json_path, images_png_path, masks_json_path, masks_png_path)

# Display the JSON format masking images
# display_json_masks(images_json_path, masks_json_path, y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
data_gen_args = dict(rotation_range=20,             # Increase rotation range
                     width_shift_range=0.1,         # Increase width shift range
                     height_shift_range=0.1,        # Increase height shift range
                     shear_range=0.1,               # Increase shear range
                     zoom_range=0.1,                # Increase zoom range
                     brightness_range=(0.9, 1.1),
                     horizontal_flip=True,
                     vertical_flip=True,            # Add vertical flipping
                     fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_datagen.fit(X_train, augment=True, seed=42)
mask_datagen.fit(y_train, augment=True, seed=42)

model = build_unet()
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

# Compile the model
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[dice_coefficient])

# Train the model
batch_size = 8  

train_generator = zip(image_datagen.flow(X_train, batch_size=batch_size, seed=42),
                      mask_datagen.flow(y_train, batch_size=batch_size, seed=42))

val_generator = zip(image_datagen.flow(X_val, batch_size=batch_size, seed=42),
                    mask_datagen.flow(y_val, batch_size=batch_size, seed=42))

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

model.fit(train_generator, steps_per_epoch=len(X_train) // batch_size, validation_data=val_generator,
          validation_steps=len(X_val) // batch_size, epochs=1)


additional_input = True  
evaluation_images = load_evaluation_images(evaluation_path, additional_input)

# Predict and display results
predicted_masks = model.predict(evaluation_images)
def convert_image_for_display(image):
    return np.uint8(image[:, :, :3])

# Set a threshold value
threshold = 0.6
num_clusters = 8

# Apply threshold to the predicted masks
binary_masks = (predicted_masks > threshold).astype(np.uint8) * 255

for i, (image, predicted_mask) in enumerate(zip(evaluation_images, predicted_masks)):
    binary_mask = (predicted_mask.squeeze() > 0.5).astype(np.uint8) * 255
    wound_area = extract_wound_area(image, binary_mask)
    
    display_image = convert_image_for_display(image)
    # display_coin_detection(display_image, best_circle, wound_area=wound_area)#draw


# coin size
COIN_DIAMETER_MM = 28.0  # Diameter of a 2-dollar coin in millimeters


image_path = evaluation_path  
image = cv2.imread(image_path)

# Detect percentage of each colour
for i, binary_mask in enumerate(binary_masks):
    original_image = convert_image_for_display(evaluation_images[i])
    masked_image = extract_color_information(original_image, binary_mask[:, :, 0])

    quantized_masked_image, centers = quantize_image(masked_image, num_clusters)
    color_percentages = calculate_color_percentage(quantized_masked_image, centers)

    print(f"Color information and their percentage for image {i}:")
    for color_info in color_percentages:
        print(f"{color_info[0]} {color_info[1]:.2f}%")

# Display the binary masks (Learning results)
for i in range(len(evaluation_images)):
    display_image = convert_image_for_display(evaluation_images[i])
    cv2.imshow(f'Original Image {i}', display_image)
    cv2.imshow(f'Binary Mask {i}', binary_masks[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save the model
model.save('wound_segmentation_model.h5')


#coin related scale wound area
image_test = cv2.imread('coin.jpg')

#pixel coount of the wound
wound_pixel = process_images(image_test)
# print(wound_pixel)

# Detect the 2-dollar coin
best_circle = detect_coin(image_test, min_radius=30, max_radius=100) #should be put in front of line 104?

# Get the actual pixel of wound area
coin_actual_area = calculate_actual_coin_area(COIN_DIAMETER_MM)
ratio_coin = best_circle[1]
ratio_wound = calculate_ratio_wound_image(wound_pixel,image_test)
calculate_actual_wound_area(ratio_coin, ratio_wound, coin_actual_area)


