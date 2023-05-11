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
import concurrent.futures
import joblib
import json
import shutil
import albumentations as A
from albumentations import Compose, ElasticTransform


evaluation_path = 'fake_evaluation/'

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
    
# Remove padding to output mask to store original image size mask
def remove_padding(image, original_width, original_height):
    height, width = image.shape[:2]

    if original_height > original_width:
        resize_ratio = height / original_height 
        resize_width = int(resize_ratio * original_width)
        left_pad = (width - resize_width) // 2
        right_pad = width - resize_width - left_pad
        top_pad =0
        bottom_pad = 0
    else:
        resize_ratio = width / original_width 
        resize_height = int(resize_ratio * original_height)
        top_pad = (height - resize_height) // 2
        bottom_pad = height - resize_height - top_pad
        left_pad = 0
        right_pad = 0

    if image.ndim == 3:
        return image[top_pad:height - bottom_pad, left_pad:width - right_pad, :]
    elif image.ndim == 2:
        return image[top_pad:height - bottom_pad, left_pad:width - right_pad]
    else:
        raise ValueError("The input image must have 2 or 3 dimensions.")


def resize_to_original(image, original_width, original_height):
    return cv2.resize(image, (original_width, original_height))



cache_folder = "cache"
    
def load_images_and_masks_worker(file, images_json_path, masks_json_path, target_size):
    cache_file = os.path.join(cache_folder, f"{file[:-5]}.joblib")
    
    if os.path.exists(cache_file):
        img, mask = joblib.load(cache_file)
    else:
        img, mask = load_image_and_mask(file, images_json_path, masks_json_path)
        if img is not None and mask is not None:
            img, mask = resize_and_pad_image_and_mask(img, mask, target_size)
            joblib.dump((img, mask), cache_file)
    
    return img, mask, file

    
def load_image_and_mask(file, images_json_path, masks_json_path):
    with open(os.path.join(masks_json_path, file[:-5] + '.json')) as f:
        mask_json = json.load(f)

    if 'Label' in mask_json:
        image = cv2.imread(os.path.join(images_json_path, file[:-5] + '.jpg'), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Unable to read image file {file}. Skipping...")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        polygon = mask_json['Label']['objects'][0]['instanceURI']
        response = requests.get(polygon)
        mask = np.array(Image.open(BytesIO(response.content)).convert('L'))

        return image, mask
    return None, None

def resize_and_pad_image_and_mask(image, mask, target_size):
    image = resize_with_aspect_ratio(image, target_size)
    image = pad_image(image, target_size)

    mean_pixel_value = np.mean(image, axis=-1, keepdims=True)
    image = np.concatenate([image, mean_pixel_value], axis=-1)

    mask = resize_with_aspect_ratio(mask, target_size)
    mask = pad_image(mask, target_size)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / 255.0

    return image, mask

def load_images_and_masks(images_json_path, masks_json_path, n_workers=4):
    images = []
    masks = []

    json_files = os.listdir(masks_json_path)
    json_files = sorted(json_files, key=lambda x: int(re.search(r'\d+', x).group()))

    target_size = 256

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(load_images_and_masks_worker, file, images_json_path, masks_json_path, target_size) for file in json_files]

        for future in concurrent.futures.as_completed(futures):
            img, mask, file = future.result()
            if img is not None and mask is not None:
                images.append(img)
                masks.append(mask)
                # print(f"Loaded image {file[:-5]}.jpg with mask {file[:-5]}.json")


    max_width = max([img.shape[1] for img in images])
    max_height = max([img.shape[0] for img in images])

    images_array = np.zeros((len(images), max_height, max_width, 4), dtype=np.float32)
    masks_array = np.zeros((len(masks), max_height, max_width, 1), dtype=np.float32)

    for i, (image, mask) in enumerate(zip(images, masks)):
        height, width = image.shape[:2]
        images_array[i, :height, :width] = image
        masks_array[i, :height, :width] = mask

    return images_array, masks_array


def augment_data(images, masks, batch_size, image_datagen, mask_datagen):
    while True:
        idx = np.random.permutation(images.shape[0])
        images = images[idx]
        masks = masks[idx]
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_masks = masks[i:i+batch_size]
            
            # Apply image_datagen and mask_datagen augmentations
            batch_images = image_datagen.flow(batch_images, batch_size=batch_size, seed=42).next()
            batch_masks = mask_datagen.flow(batch_masks, batch_size=batch_size, seed=42).next()
            
            aug_images = []
            aug_masks = []
            
            for img, mask in zip(batch_images, batch_masks):
                # Apply Albumentations library augmentations
                augmented = Compose([
                    ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
                    # A.RandomContrast(limit=0.2, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    # A.RandomCrop(height=128, width=128, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A. GaussNoise(var_limit=(10, 50), p=0.5),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5)
                ])(image=img, mask=mask)
                
                aug_images.append(augmented['image'])
                aug_masks.append(augmented['mask'])
                

            yield np.array(aug_images), np.array(aug_masks)



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
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image file {image_path}. Check file path/integrity")
    
    # Get the dimensions of the original image
    original_height, original_width = image.shape[:2]

    # Determine whether the original image's height or width is bigger
    if original_height > original_width:
        long_side = original_height
        short_side = original_width
    else:
        long_side = original_width
        short_side = original_height

    # Resize the image while maintaining its aspect ratio
    target_size = 256
    image = resize_with_aspect_ratio(image, target_size)
    image = pad_image(image, target_size)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper boundaries for the blue color
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask that isolates the blue color in the image
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Perform contour detection
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the largest contour on a white background
    contour_image = np.zeros_like(image)
    contour_image.fill(255)
    cv2.drawContours(contour_image, contours, -1, (0, 0, 0), 2)

    # Create a binary mask where the contour with wound region is white (255) and the rest of the image is black (0)
    wound_area = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(wound_area, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Count the pixels inside the filled contour
    pixel_count = np.count_nonzero(wound_area)

    # Get the ratio of the resized image to origical image
    resize_ratio = 256 / long_side 
    resize_short_size = resize_ratio * short_side


    # Calculate the total number of pixels in the resized image
    total_pixels = 256 * resize_short_size

    # Calculate the ratio of pixels inside the filled contour to the total pixels in the resized image
    pixel_ratio = pixel_count / total_pixels


    return contour_image, pixel_count, pixel_ratio, wound_area



def process_images(directory):
    # Get all the files in the directory
    files = os.listdir(directory)

    # Filter out files that are not images (assuming .jpg format)
    image_files = [file for file in files if file.lower().endswith(".jpg")]

    # Sort image files numerically
    image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))

    pixel_counts = {}
    pixel_ratios = {}

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        pixel_count = extract_blue_contour(image_path)[1]
        pixel_ratio = extract_blue_contour(image_path)[2]
        print(f"Number of pixels inside the contour for {image_file}: {pixel_count}\n")
        
        print(f"Ratio of pixels inside the contour for {image_file}: {pixel_ratio}")

        # cv2.imshow(f"Contour Image for {image_file}", contour_image)
        cv2.waitKey(0)

        pixel_counts[image_file] = pixel_count
        pixel_ratios[image_file] = pixel_ratio

    cv2.destroyAllWindows()

    return pixel_counts, pixel_ratios

def process_image(image,image_path):
    # image_path = os.path(image)
    pixel_ratio = extract_blue_contour(image_path)[2]

    cv2.destroyAllWindows()

    return pixel_ratio


def split_json_objects(input_file, output_folder):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_number = 371

    for obj in data:
        output_file = os.path.join(output_folder, f"{file_number}.json")
        with open(output_file, 'w') as outfile:
            json.dump(obj, outfile)
        file_number += 1



