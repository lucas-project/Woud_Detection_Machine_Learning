import os
import json
import numpy as np
import cv2
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
import re
import matplotlib.pyplot as plt
from skimage import measure

# Resize while keeping original ratio (Avoid coin shape change)
def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > h:
        new_width = target_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(new_height * aspect_ratio)

    return cv2.resize(image, (new_width, new_height))

# Add padding to output image to store resized image
def pad_image(image, target_size):
    height, width = image.shape[:2]
    new_width, new_height = target_size, target_size

    top_pad = (new_height - height) // 2
    bottom_pad = new_height - height - top_pad
    left_pad = (new_width - width) // 2
    right_pad = new_width - width - left_pad

    if image.ndim == 3:
        return np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    elif image.ndim == 2:
        return np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
    else:
        raise ValueError("The input image must have 2 or 3 dimensions.")
    
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
            target_size = 256

            image = cv2.imread(os.path.join(images_json_path, file[:-5] + '.jpg'), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Unable to read image file {file}. Skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_with_aspect_ratio(image, target_size)
            image = pad_image(image, target_size)

            # Add additional channel
            mean_pixel_value = np.mean(image, axis=-1, keepdims=True)
            image = np.concatenate([image, mean_pixel_value], axis=-1)
            
            images.append(image)

            polygon = mask_json['Label']['objects'][0]['instanceURI']
            response = requests.get(polygon)
            mask = np.array(Image.open(BytesIO(response.content)).convert('L'))
            mask = resize_with_aspect_ratio(mask, target_size)
            mask = pad_image(mask, target_size)
            mask = np.expand_dims(mask, axis=-1)  # Convert to (256, 256, 1) shape
            mask = mask / 255.0  # Scale mask values between 0 and 1
            masks.append(mask)
            print(f"Loaded JSON mask for file {file}")

    # Load images with PNG masks
    png_files = os.listdir(images_png_path)

    # Sort PNG files numerically
    png_files = sorted(png_files, key=lambda x: int(re.search(r'\d+', x).group()))

    for file in png_files:
        # Load PNG mask
        mask = cv2.imread(os.path.join(masks_png_path, file[:-4] + '.png'), cv2.IMREAD_GRAYSCALE)

        image = cv2.imread(os.path.join(images_png_path, file), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Unable to read image file {file}. Skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_with_aspect_ratio(image, target_size)

        # Add additional channel
        mean_pixel_value = np.mean(image, axis=-1, keepdims=True)
        image = np.concatenate([image, mean_pixel_value], axis=-1)
        image = pad_image(image, target_size)

        mask = resize_with_aspect_ratio(mask, target_size)
        mask = pad_image(mask, target_size)
        mask = np.expand_dims(mask, axis=-1)  # Convert to (256, 256, 1) shape
        mask = mask / 255.0  # Scale mask values between 0 and 1

        images.append(image)
        masks.append(mask)
        print(f"Loaded PNG mask for file {file}")

        # height of the images and masks
    max_width = max([img.shape[1] for img in images])
    max_height = max([img.shape[0] for img in images])

    # Create empty NumPy arrays with a consistent shape
    images_array = np.zeros((len(images), max_height, max_width, 4), dtype=np.float32)
    masks_array = np.zeros((len(masks), max_height, max_width, 1), dtype=np.float32)

    # Fill the empty NumPy arrays with the resized images and masks
    for i, (image, mask) in enumerate(zip(images, masks)):
        height, width = image.shape[:2]
        images_array[i, :height, :width] = image
        masks_array[i, :height, :width] = mask

    return images_array, masks_array


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

# Display JSON masking images (Check whether masking JSON file has been read by model correctly)
# def display_json_masks(masks_json_path, masks):
#     for idx, file in enumerate(os.listdir(masks_json_path)):
#         with open(os.path.join(masks_json_path, file)) as f:
#             mask_json = json.load(f)

#         if 'Label' in mask_json and 'objects' in mask_json['Label']:
#             instance_uri = mask_json['Label']['objects'][0]['instanceURI']
#             response = requests.get(instance_uri)
#             mask = np.array(Image.open(BytesIO(response.content)).convert('L'))
#             mask = resize_with_aspect_ratio(mask, 256)
#         else:
#             print(f"No 'Label' or 'objects' key found for {file}. Skipping...")
#             continue

#         cv2.imshow(file, mask)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
import re

def display_json_masks(images_json_path, masks_json_path, masks):
    json_files = os.listdir(masks_json_path)
    # Sort JSON files numerically
    json_files = sorted(json_files, key=lambda x: int(re.search(r'\d+', x).group()))

    for idx, file in enumerate(json_files):
        with open(os.path.join(masks_json_path, file)) as f:
            mask_json = json.load(f)

        if 'Label' in mask_json and 'objects' in mask_json['Label']:
            instance_uri = mask_json['Label']['objects'][0]['instanceURI']
            response = requests.get(instance_uri)
            mask = np.array(Image.open(BytesIO(response.content)).convert('L'))
            mask = resize_with_aspect_ratio(mask, 256)

            # Load the corresponding image
            image = cv2.imread(os.path.join(images_json_path, file[:-5] + '.jpg'), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Unable to read image file {file}. Skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_with_aspect_ratio(image, 256)
        else:
            print(f"No 'Label' or 'objects' key found for {file}. Skipping...")
            continue

        # Ensure the image and mask have the same dimensions
        image = pad_image(image, 256)
        mask = pad_image(mask, 256)

        # Add an extra channel to the mask and repeat it along the channel axis
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)  # Repeat the mask along the channel axis to match the image

        # Display image and mask side by side
        image_file_name = file[:-5] + '.jpg'
        json_file_name = file
        window_title = f"Image: {image_file_name} | JSON: {json_file_name}"
        combined = np.concatenate((image, mask), axis=1)
        cv2.imshow(window_title, combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





# Get wound area from evaluation dataset
def extract_wound_area(image, binary_mask):
    # Apply morphological dilation to the predicted mask
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    # Find contours in the dilated mask
    contours = measure.find_contours(dilated_mask, 0.5)

    # Find the largest contour
    max_contour = max(contours, key=lambda x: len(x))

    # Create an array of tuples containing the x and y coordinates of each point in the contour
    wound_area = np.array([(int(point[1]), int(point[0])) for point in max_contour])

    return wound_area


def extract_blue_contour(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper boundaries for the blue color
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Apply a mask to extract the blue color
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Apply dilation and erosion to reduce noise and improve the contour
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Find the contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new white image with the same size as the original image
    contour_image = np.ones(image.shape, dtype=np.uint8) * 255

    # Draw the contours on the new white image
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

    return contour_image