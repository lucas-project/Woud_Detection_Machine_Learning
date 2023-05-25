import os
import numpy as np
import json
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import re
import matplotlib.pyplot as plt
from skimage import measure
from v1_border import data_gen_args, build_unet, display_json_masks, extract_wound_area, load_images_and_masks, split_json_objects, augment_data, resize_to_original, remove_padding, fine_tune_model, convert_image_for_display
from v1_coin import detect_coin
from v1_processing import extract_contours_from_outlined_image
from v1_measurement import calculate_pixels_per_millimetre_ratio, get_circle_area_px, get_circle_area_mm, get_contour_size_px, get_contour_area_px, get_contour_size_mm, get_contour_area_mm
from v1_visualisation import visualise_circle_area_mm, visualise_circle_radius_mm, visualise_circle_diameter_mm, visualise_contour_area_mm, visualise_contour_size_mm
from v1_comparison import save_wound_data, load_wound_data, plot_wound_data
from v1_colour import calculate_color_percentage, quantize_image, extract_color_information
from v1_evaluation import load_evaluation_images, load_fake_evaluation_images
import time
from keras.models import load_model

# Set up path
images_json_path = 'fake_wound/'
masks_json_path = 'fake_jj/'
evaluation_path = 'fake_evaluation/'
#input_directory = 'contour/' # Only used for old contour extraction function
#input_file = 'splited_json/1.json'
#output_folder = 'splited_json/'
model_path = 'models/model_finetuned_1683628583_5_0.0002.h5'

batch_size = 7

# This function only used when need to split .json file from labelbox to small .json file,  
# file name needed to changed each time to generate correct file name.

# split_json_objects(input_file, output_folder)

# Coin radius
COIN_RADIUS_MM = 10.25 # The radius of an Australian $2 coin in millimetres

# The path of the image for analysis
# TODO: Should we change this to accept a user input?
input_image_path = 'healing/1.jpg'
output_result_path = 'results'

### LOAD/PREPROCESS IMAGES ###

# Load the image for measurements
input_image = cv2.imread(input_image_path)

### MEASUREMENTS ###

## Coin Measurement 

# Get the radius of the found coin in pixels
coin_circle, _ = detect_coin(input_image)
coin_position_x, coin_position_y, coin_radius_px = coin_circle

# Calculate the pixels-per-millimetre ratio
pixels_per_millimetre_ratio = calculate_pixels_per_millimetre_ratio(coin_radius_px, COIN_RADIUS_MM)

# Get the area of the coin in pixels^2
coin_area_px = get_circle_area_px(coin_radius_px)

# Get the area of the coin in millimetres^2
coin_area_mm = get_circle_area_mm(coin_area_px, pixels_per_millimetre_ratio)

# Print coin measurements
print()
print(f'Pixels-per-millimetre Ratio: {pixels_per_millimetre_ratio}')
print()
print('Coin measurements:')
print(f'  Coin Radius: {coin_radius_px:.2f}px, {COIN_RADIUS_MM:.2f}mm')
print(f'  Coin Area: {coin_area_px:.2f}px^2, {coin_area_mm:.2f}mm^2')
print()

## Wound Measurement ##

# Extract areas from the input image which have blue outlines drawn around them
wound_contours = extract_contours_from_outlined_image(input_image)

# Make a copies of the original image for visualising measurements
image_areas = input_image.copy()
image_lengths = input_image.copy()

# Create an empty array to store wound measurements for saving
wound_results = []

# Loop through and measure each wound
for i, wound_contour in enumerate(wound_contours):
	# Wound measurements in pixels
	wound_area_px = get_contour_area_px(wound_contour)
	wound_length_x_px, wound_length_y_px = get_contour_size_px(wound_contour)

	# Convert the wound measurements to millimetres
	wound_area_mm = get_contour_area_mm(wound_contour, pixels_per_millimetre_ratio)
	wound_length_x_mm, wound_length_y_mm = get_contour_size_mm(wound_contour, pixels_per_millimetre_ratio)

	# Print wound measurements
	wound_name = f'Wound {i}'
	
	#print()
	print(f'Measurements for {wound_name}:')
	print(f'  Wound Length X: {wound_length_x_px:.2f}px, {wound_length_x_mm:.2f}mm')
	print(f'  Wound Length Y: {wound_length_y_px:.2f}px, {wound_length_y_mm:.2f}mm')
	print(f'  Wound Area: {wound_area_px:.2f}px^2, {wound_area_mm:.2f}mm^2')
	print()

	# Store the results
	wound_results.append([wound_length_x_mm, wound_length_y_mm, wound_area_mm])

	### VISUALISATIONS ###

	## Visualise Areas ##

	# Draw coin area visualisation
	image_areas = visualise_circle_area_mm(image_areas, coin_circle, coin_area_mm, name='Scale Ref.')

	# Draw wound area visualisation
	image_areas = visualise_contour_area_mm(image_areas, wound_contour, wound_area_mm, name=wound_name)

	## Visualise Lengths ##

	# Draw coin length visualisation
	image_lengths = visualise_circle_diameter_mm(image_lengths, coin_circle, COIN_RADIUS_MM * 2, name='Scale Ref.')

	# Draw wound lengths visualisation
	image_lengths = visualise_contour_size_mm(image_lengths, wound_contour, wound_length_x_mm, wound_length_y_mm, name=wound_name)

## Display Visualisations ##

# Display visualisations
cv2.imshow('Original Image', input_image)
cv2.imshow('Area Measurements', image_areas)
cv2.imshow('Length Measurements', image_lengths)

# Convert contours to a binary mask and display it (comment out if not needed)
result_mask = np.zeros_like(input_image[:,:,0])
cv2.drawContours(result_mask, wound_contours, -1, 255, -1)
cv2.imshow('Wound Mask', result_mask)

# Wait for keypress, then close all image windows
cv2.waitKey(0)
cv2.destroyAllWindows()

### SAVE RESULTS ###

if wound_results:
	print()
	print('Do you wish to save these results? (Y/N)')
	
	input_save = input()

	if input_save.lower() == 'y':
		# Prompt used for a name to use for the results
		output_name = None
		
		# Continue asking until a valid name is given (not None)
		while not output_name:
			print()
			print('Please enter a name:')
			output_name = input()
			
			if not output_name:
				print('No name detected. A valid name must be used.')
		
		# Save the results
		save_wound_data(output_result_path, output_name, input_image, result_mask, wound_results)

### COMPARE RESULTS ###

print()
print('Would you like to load a patients result history? (Y/N)')

input_load = input()

if input_load.lower() == 'y':
	# Prompt the user for a name to load the results for
	input_load = None
	
	# Continue asking until a valid name is given (not None)
	while not input_load:
		print()
		print('Please type the patients name:')
		input_load = input()
		
		if not input_load:
			print('No name detected. A valid name must be used.')
	
	# Get the path the results should be in
	result_path = os.path.join(output_result_path, input_load)
	
	# Load the results
	results = load_wound_data(result_path)
	
	# Check to see if results were loaded correctly (not None)
	if results:
		# If results exist, plot them
		plot_wound_data(results)

print()
print("Do you wish to continue to model training? (Y/N)")
input_continue = input()

if input_continue.lower() != 'y':
	quit()

### AI MODEL ###

# load_images_and_masks
X, y = load_images_and_masks(images_json_path, masks_json_path)

# Display the JSON format masking images
# display_json_masks(images_json_path, masks_json_path, y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_datagen.fit(X_train, augment=True, seed=42)
mask_datagen.fit(y_train, augment=True, seed=42)


def dice_coefficient(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

load_old_model = input("Press any key to load an old model, or type 'n' to train a new model: ")

if load_old_model.lower() != 'n' and os.path.exists(model_path):
    # Load the saved model
    print("Loading saved model...")
    model = load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})
else:
    # Train a new model
    print("Training a new model...")
    model = build_unet()
    
    # Compile the model, learning rate default is 0.001
    optimizer = Adam(learning_rate = 0.0007)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=[dice_coefficient])


    # Train the model

    train_generator = augment_data(X_train, y_train, batch_size, image_datagen, mask_datagen)
    val_generator = augment_data(X_val, y_val, batch_size, image_datagen, mask_datagen)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    history = model.fit(train_generator, steps_per_epoch=len(X_train) // batch_size, validation_data=val_generator,
              validation_steps=len(X_val) // batch_size, epochs=15)

    # Save the trained model
    model.save(model_path)

if load_old_model.lower() == 'n':
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

additional_input = True  
evaluation_images, original_dimensions, original_images = load_fake_evaluation_images(evaluation_path, additional_input)

# Predict and display results
predicted_masks = model.predict(evaluation_images)


# Set a threshold value
threshold = 0.6
num_clusters = 8

# Apply threshold to the predicted masks
binary_masks = (predicted_masks > threshold).astype(np.uint8) * 255

for i, (image, predicted_mask) in enumerate(zip(evaluation_images, predicted_masks)):
    binary_mask = (predicted_mask.squeeze() > 0.5).astype(np.uint8) * 255
    wound_area = extract_wound_area(image, binary_mask)
    
    display_image = convert_image_for_display(image)

    # Detect the 2-dollar coin for each image
    # best_circle, ratio_coin = detect_coin(display_image)

    # Display the coin detection result
    # display_coin_detection(display_image, best_circle, wound_area=wound_area)


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
    display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    cv2.imshow(f'Original Image {i}', display_image)
    cv2.imshow(f'Binary Mask {i}', binary_masks[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


resized_binary_masks = []
for i, (binary_mask, original_dim) in enumerate(zip(binary_masks, original_dimensions)):
    original_height, original_width = original_dim
    remove_padding_mask = remove_padding(binary_mask, original_width, original_height)
    resized_mask = resize_to_original(remove_padding_mask, original_width, original_height)
    resized_binary_masks.append(resized_mask)
    # display_image = convert_image_for_display(original_images[i])
    # cv2.imshow(f'Original Image {i}', display_image)
    # cv2.imshow(f'Resized Binary Mask {i}', resized_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


for i, binary_mask_array in enumerate(resized_binary_masks):
    mask_pixels = np.sum(binary_mask_array == 255)  # Count the number of mask pixels (value 255)
    total_pixels = binary_mask_array.size  # Calculate the total number of pixels in the binary mask
    mask_pixel_ratio = mask_pixels / total_pixels  # Calculate the mask pixel ratio

    print(f"Mask pixel ratio for image {i}: {mask_pixel_ratio:.4f}")
    print(f"Mask pixel for image {i}: {mask_pixels:.4f}")


# resized_binary_masks_pixels = []
# for resized_mask in resized_binary_masks:
#     binary_mask_pixels = np.where(resized_mask > 0)  # Get the indices where the binary mask is non-zero
#     mask_pixels = []
#     for row, col in zip(binary_mask_pixels[0], binary_mask_pixels[1]):
#         pixel_value = resized_mask[row, col]
#         mask_pixels.append((row, col, pixel_value))
#     resized_binary_masks_pixels.append(mask_pixels)
# print (resized_binary_masks_pixels)

# Save the model
if load_old_model.lower() == 'n':
    timestamp = int(time.time())
    model.save(f'models/wound_segmentation_model_{timestamp}.h5')

# Fine tune option
fine_tune_model1 = input("Press 'y' to fine-tune the model first time, or 'n' for no: ")

if fine_tune_model1.lower() == 'y':
    model = fine_tune_model(model, 'fine_tune_2/', 'fine_tune_2_masks/', original_images, resized_binary_masks)

fine_tune_model2 = input("Press 'y' to fine-tune the model second time, or 'n' for no: ")

if fine_tune_model2.lower() == 'y':
    model = fine_tune_model(model, 'fine_tune_1/', 'fine_tune_1_masks/', original_images, resized_binary_masks)







