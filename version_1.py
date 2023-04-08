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
images_json_path = 'C:/Users/User/Desktop/ml/fake_wound/'
masks_json_path = 'C:/Users/User/Desktop/ml/fake_jj/'
# images_png_path = 'C:/Users/User/Desktop/ml/png_images/'
# masks_png_path = 'C:/Users/User/Desktop/ml/png_masking/'
images_png_path = 'C:/Users/User/Desktop/ml/fake_png_1/'
masks_png_path = 'C:/Users/User/Desktop/ml/fake_png_2/'
evaluation_path = 'C:/Users/User/Desktop/ml/fake_evaluation/'

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
            if image is None:
                print(f"Unable to read image file {file}. Skipping...")
                continue
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
        if image is None:
            print(f"Unable to read image file {file}. Skipping...")
            continue
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
          validation_steps=len(X_val) // batch_size, epochs=5)

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

def extract_wound_area(image, binary_mask):
    wound_area = cv2.bitwise_and(image, image, mask=binary_mask)
    return wound_area


# Set a threshold value
threshold = 0.6
num_clusters = 8


# Apply threshold to the predicted masks
binary_masks = (predicted_masks > threshold).astype(np.uint8) * 255

def calculate_color_percentage(image, centers):
    unique_colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    color_percentages = []

    for color, count in zip(unique_colors, counts):
        color_hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        percentage = (count / np.sum(counts)) * 100
        color_percentages.append((color_hex, percentage))

    return color_percentages

def quantize_image(image, num_clusters=8):
    pixels = image.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    quantized_image = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    return quantized_image, centers

def extract_color_information(original_image, mask):
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    return masked_image


# Constants
COIN_DIAMETER_MM = 28.0  # Diameter of a 2-dollar coin in millimeters

# Function to detect the 2-dollar coin using the Hough Circle Transform
def detect_coin(image, min_radius, max_radius):
    if image is None:
        print("Error loading the image.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=30, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[0]  # Return the first detected circle
    else:
        return None


# Function to calculate the wound area in pixels
def calculate_wound_area(mask):
    return np.sum(mask == 255)

# Function to estimate actual wound area using scaling factor
def estimate_actual_area(pixels_area, scaling_factor):
    return pixels_area * scaling_factor * scaling_factor

# Main code
image_path = evaluation_path  
image = cv2.imread(image_path)

# Detect the 2-dollar coin
coin = detect_coin(image, min_radius=30, max_radius=100)

if coin is not None:
    # Calculate the diameter of the coin in the image (in pixels)
    coin_diameter_pixels = coin[2] * 2

    # Calculate the scaling factor (actual diameter / image diameter)
    scaling_factor = COIN_DIAMETER_MM / coin_diameter_pixels

    # Calculate the wound area in the image (in pixels)
    # Replace 'binary_mask' with the actual binary mask variable from your implementation
    wound_area_pixels = calculate_wound_area(binary_masks)

    # Estimate the actual wound area using the scaling factor
    actual_wound_area = estimate_actual_area(wound_area_pixels, scaling_factor)
    print(f"Actual wound area: {actual_wound_area:.2f} square millimeters")
else:
    print("No 2-dollar coin detected. Cannot estimate actual wound area.")



for i, binary_mask in enumerate(binary_masks):
    original_image = convert_image_for_display(evaluation_images[i])
    masked_image = extract_color_information(original_image, binary_mask[:, :, 0])

    quantized_masked_image, centers = quantize_image(masked_image, num_clusters)
    color_percentages = calculate_color_percentage(quantized_masked_image, centers)

    print(f"Color information and their percentage for image {i}:")
    for color_info in color_percentages:
        print(f"{color_info[0]} {color_info[1]:.2f}%")


# Display the binary masks
for i in range(len(evaluation_images)):
    display_image = convert_image_for_display(evaluation_images[i])
    cv2.imshow(f'Original Image {i}', display_image)
    cv2.imshow(f'Binary Mask {i}', binary_masks[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Save the model
model.save('wound_segmentation_model.h5')


