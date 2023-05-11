import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
import glob
from v1_evaluation import load_evaluation_images
from version_1 import dice_coefficient

evaluation_path = 'more_evaluation/'


# Get the latest model file in the 'saved_model' folder
list_of_model_files = glob.glob('saved_model/*.h5')
latest_model_file = max(list_of_model_files, key=os.path.getctime)

# Load the latest model
loaded_model = load_model(latest_model_file, custom_objects={'dice_coefficient': dice_coefficient})

# Load the saved model
loaded_model = load_model('saved_model/wound_segmentation_model.h5', custom_objects={'dice_coefficient': dice_coefficient})

# Load and preprocess new dataset
additional_input = True 
evaluation_images = load_evaluation_images(evaluation_path, additional_input)  # You need to implement this function to load and preprocess the new dataset

# Predict using the loaded model
predicted_masks = loaded_model.predict(evaluation_images)
threshold = 0.6
binary_masks = (predicted_masks > threshold).astype(np.uint8) * 255


# Evaluate, display results, or perform other necessary operations using the predictions
predicted_masks = loaded_model.predict(evaluation_images)


def convert_image_for_display(image):
    return np.uint8(image[:, :, :3])

for i in range(len(evaluation_images)):
    display_image = convert_image_for_display(evaluation_images[i])
    cv2.imshow(f'Original Image {i}', display_image)
    cv2.imshow(f'Predicted Mask {i}', binary_masks[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()