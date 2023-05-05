import os
import numpy as np
import cv2
import re
from v1_coin import detect_coin, display_coin_detection
from v1_border import resize_with_aspect_ratio, pad_image

# Evaluate the model's performance on the evaluation set
def load_evaluation_images(evaluation_path, additional_input):
    min_radius = 10
    max_radius = 50
    images = []
    evaluation_images = []
    image_files = os.listdir(evaluation_path)
    image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))

    for file in image_files:
        image = cv2.imread(os.path.join(evaluation_path, file), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Unable to read image file {file}. Skipping...")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_with_aspect_ratio(image, 256)
        image = pad_image(image, 256)

        if additional_input:  # Use the boolean value to conditionally add the additional channel
            mean_pixel_value = np.mean(image, axis=-1, keepdims=True)
            image = np.concatenate([image, mean_pixel_value], axis=-1)

        evaluation_images.append(image)
        print(f"Loaded evaluation image {file}")

        # Detect the coin in the image
        # coin_detected = detect_coin(image, min_radius, max_radius)[0]

        # display_coin_detection(image, coin_detected, None)

    # Find the maximum width and height of the evaluation images
    max_width = max([img.shape[1] for img in evaluation_images])
    max_height = max([img.shape[0] for img in evaluation_images])

    # Create an empty NumPy array with a consistent shape
    evaluation_images_array = np.zeros((len(evaluation_images), max_height, max_width, 4), dtype=np.float32)

    # Fill the empty NumPy array with the resized evaluation images
    for i, image in enumerate(evaluation_images):
        height, width = image.shape[:2]
        evaluation_images_array[i, :height, :width] = image

    return evaluation_images_array