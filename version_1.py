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
#from v1_border import build_unet, display_json_masks, extract_wound_area, load_images_and_masks, extract_blue_contour, process_image, process_images, split_json_objects, augment_data, resize_to_original, remove_padding
from v1_border import build_unet, display_json_masks, extract_wound_area, load_images_and_masks, process_image, process_images, split_json_objects, augment_data, resize_to_original, remove_padding
#from v1_coin import display_coin_detection, detect_coin, calculate_actual_coin_area, calculate_ratio_wound_image, calculate_actual_wound_area
from v1_coin import detect_coin
#from v1_processing import extract_contour_from_outlined_image
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
#images_png_path = 'fake_png_1/'
#masks_png_path = 'fake_png_2/'
evaluation_path = 'fake_evaluation/'
#input_directory = 'contour/'
#output_directory = 'contour_processed/'
#input_file = 'splited_json/1.json'
#output_folder = 'splited_json/'
model_path = 'models/model_finetuned_1683628583_5_0.0002.h5'

batch_size = 7

# This function only used when need to split .json file from labelbox to small .json file,  
# file name needed to changed each time to generate correct file name.

# split_json_objects(input_file, output_folder)

## coin size
#COIN_DIAMETER_MM = 28.0  # Diameter of a 2-dollar coin in millimetres
## The above value is wrong! The diameter is 20.5mm!

#if not os.path.exists(output_directory):
#    os.makedirs(output_directory)

## pixel coount of the wound
#pixel_counts, pixel_ratios = process_images(input_directory)
#print(pixel_counts)

#for image_file in os.listdir(input_directory):
#    if image_file.endswith('.jpg'):
#        input_path = os.path.join(input_directory, image_file)
#        output_path = os.path.join(output_directory, image_file)
#        contour_image, pixel_count, pixel_ratio, wound_area = extract_blue_contour(input_path) 
#        cv2.imwrite(output_path, contour_image)
#
#        # Display the output image
#        #cv2.imshow(f'Wound Image for {image_file}', wound_area)
#        cv2.waitKey(0)

#cv2.destroyAllWindows()

##coin related scale wound area
#image_test = cv2.imread('contour/1.jpg')
#image_path = 'contour/1.jpg'
##pixel coount of the wound
#wound_ratio = process_image(image_test,image_path)
## print(wound_pixel)
#
## Detect the 2-dollar coin
#best_circle = detect_coin(image_test) #should be put in front of line 104?
#
## Get the actual pixel of wound area
#coin_actual_area = calculate_actual_coin_area(COIN_DIAMETER_MM)
#ratio_coin = best_circle[1]
## ratio_wound = calculate_ratio_wound_image(wound_pixel,image_test,image_path)
#wound_area = calculate_actual_wound_area(ratio_coin, wound_ratio, coin_actual_area)
#print(f"coin's area is :{coin_actual_area}")
#print(f"coin/image's ratio is :{ratio_coin}")
#print(f"wound/image's ratio is :{wound_ratio}")
#print(f"wound area is :{wound_area}")

# Coin radius
COIN_RADIUS_MM = 10.25 # The radius of an Australian $2 coin in millimetres

# The path of the image for analysis
# TODO: Should we change this to accept a user input?
#input_image_path = 'contour/12.jpg'
input_image_path = 'healing/3.jpg'
output_result_path = 'results'

### LOAD/PREPROCESS IMAGES ###

# Load the image for measurements
input_image = cv2.imread(input_image_path)

## Make a copy of the original image for the coin
#coin_image = original_image.copy()

#coin_image_target_size = original_image.shape[:2]

#coin_image = pad_image(coin_image, max(coin_image_target_size))

## Make a copy of the original image for the wound
#wound_image = original_image.copy()

#wound_image_target_size = 256

#wound_image = resize_with_aspect_ratio(wound_image, wound_image_target_size)
#wound_image = pad_image(wound_image, wound_image_target_size)

### MEASUREMENTS ###

## Coin Measurement ##

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

# Extract the wound as a binary mask from an image with a blue outline
#_, _, _, wound_mask = extract_blue_contour(image_path) # Please note how few return values are being used

## Extract the contour from the input image with a blue outline drawn around the wound
#wound_contour = extract_contour_from_outlined_image(input_image)
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

	## Make a copy of the original image
	#image_areas = input_image.copy()

	# Draw coin area visualisation
	image_areas = visualise_circle_area_mm(image_areas, coin_circle, coin_area_mm, name='Scale Ref.')

	# Draw wound area visualisation
	image_areas = visualise_contour_area_mm(image_areas, wound_contour, wound_area_mm, name=wound_name)

	## Visualise Lengths ##

	## Make a copy of the original image
	#image_lengths = input_image.copy()

	# Draw coin length visualisation
	# TODO: Current visualisation is diamater. Function also exists to visualise radius. Confirm which is desirable.
	image_lengths = visualise_circle_diameter_mm(image_lengths, coin_circle, COIN_RADIUS_MM * 2, name='Scale Ref.')

	# Draw wound lengths visualisation
	image_lengths = visualise_contour_size_mm(image_lengths, wound_contour, wound_length_x_mm, wound_length_y_mm, name=wound_name)

## Display Visualisations ##

# Display visualisations
cv2.imshow('Original Image', input_image)
cv2.imshow('Area Measurements', image_areas)
cv2.imshow('Length Measurements', image_lengths)

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
		
		# Convert contour to binary mask
		result_mask = np.zeros_like(input_image[:,:,0])
		cv2.drawContours(result_mask, wound_contours, -1, 255, -1)
		
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
# TODO: This function fails for me (Drew) due to some kind of authentication error.
# I cannot find any documentation in the project about this, nor was any communicated to me
# so I am commenting it out for now. I will try to remember to ask about it next time we meet.
#display_json_masks(images_json_path, masks_json_path, y)

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
    optimizer = Adam(learning_rate = 0.0005)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=[dice_coefficient])


    # Train the model


    

    train_generator = augment_data(X_train, y_train, batch_size, image_datagen, mask_datagen)
    val_generator = augment_data(X_val, y_val, batch_size, image_datagen, mask_datagen)


# train_generator = zip(image_datagen.flow(X_train, batch_size=batch_size, seed=42),
#                       mask_datagen.flow(y_train, batch_size=batch_size, seed=42))

# val_generator = zip(image_datagen.flow(X_val, batch_size=batch_size, seed=42),
#                     mask_datagen.flow(y_val, batch_size=batch_size, seed=42))



    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    history = model.fit(train_generator, steps_per_epoch=len(X_train) // batch_size, validation_data=val_generator,
              validation_steps=len(X_val) // batch_size, epochs=8)

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
fine_tune_model = input("Press 'y' to fine-tune the model, or 'n' for no: ")

if fine_tune_model.lower() == 'y':
    # Make some layers trainable, for example, the last 5 layers
    for layer in model.layers[-5:]:
        layer.trainable = True

    new_images_json_path = 'fine_tune_1/'
    new_masks_json_path = 'fine_tune_1_masks/'

    # Compile the model with a potentially different learning rate
    optimizer = Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=[dice_coefficient])

    # Load the new dataset images and masks
    X_new, y_new = load_images_and_masks(new_images_json_path, new_masks_json_path)

    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

    image_datagen_new = ImageDataGenerator(**data_gen_args)
    mask_datagen_new = ImageDataGenerator(**data_gen_args)

    image_datagen_new.fit(X_train_new, augment=True, seed=42)
    mask_datagen_new.fit(y_train_new, augment=True, seed=42)

    train_generator_new = zip(image_datagen_new.flow(X_train_new, batch_size=batch_size, seed=42),
                              mask_datagen_new.flow(y_train_new, batch_size=batch_size, seed=42))

    val_generator_new = zip(image_datagen_new.flow(X_val_new, batch_size=batch_size, seed=42),
                            mask_datagen_new.flow(y_val_new, batch_size=batch_size, seed=42))

    history = model.fit(train_generator_new, steps_per_epoch=len(X_train_new) // batch_size, validation_data=val_generator_new,
              validation_steps=len(X_val_new) // batch_size, epochs=8)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()

    # Save the fine-tuned model
    timestamp = int(time.time())
    # Load evaluation images and display them side by side with their masks
    evaluation_images, original_dimensions, original_images = load_fake_evaluation_images(evaluation_path, additional_input)

    # Predict and display results
    predicted_masks = model.predict(evaluation_images)

    # ... Existing code for processing and displaying the predicted masks ...

    for i, (original_image, resized_mask) in enumerate(zip(original_images, resized_binary_masks)):
        display_image = convert_image_for_display(original_image)
        cv2.imshow(f'Original Image {i}', display_image)
        cv2.imshow(f'Resized Binary Mask {i}', resized_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    model.save(f'models/model_finetuned_{timestamp}.h5')







