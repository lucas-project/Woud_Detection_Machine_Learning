import math
import cv2

from v1_border import extract_blue_contour, resize_with_aspect_ratio, pad_image
from v1_coin import detect_coin

from v1_measurement import calculate_pixels_per_millimetre_ratio, pixels_to_millimetres, millimetres_to_pixels, get_circle_area_px, get_circle_area_mm, get_contour_size_px, get_contour_area_px, get_contour_size_mm, get_contour_area_mm
from v1_visualisation import visualise_circle_area_mm, visualise_circle_radius_mm, visualise_circle_diameter_mm, visualise_contour_area_mm, visualise_contour_size_mm

image_path = 'contour/1.jpg'

image = cv2.imread(image_path)

### COIN MEASUREMENT ###

# The radius of an Australian $2 coin in millimetres
coin_radius_mm = 10.25

# Get the radius of the found coin in pixels
best_circle, _ = detect_coin(image)
coin_position_x, coin_position_y, coin_radius_px = best_circle

# Calculate the pixels-per-millimetre ratio
pixels_per_millimetre_ratio = calculate_pixels_per_millimetre_ratio(coin_radius_px, coin_radius_mm)

coin_area_px = get_circle_area_px(coin_radius_px)
coin_area_mm = get_circle_area_mm(coin_area_px, pixels_per_millimetre_ratio)
# This is just to validate that the conversion works the other way as well
coin_area_px = millimetres_to_pixels(coin_area_mm, pixels_per_millimetre_ratio, 2)

print()
print(f'Pixels-per-millimetre Ratio: {pixels_per_millimetre_ratio}')
print()
print(f'Coin Radius: {"{:.2f}px".format(coin_radius_px)}, {"{:.2f}mm".format(coin_radius_mm)}')
print(f'Coin Area: {"{:.2f}px^2".format(coin_area_px)}, {"{:.2f}mm^2".format(coin_area_mm)}')
print()
print(f'Coin Area Validated: {True if round(coin_area_mm, 2) == 330.06 else False} (expected: 330.06mm^2)')

### WOUND MEASUREMENT ###

_, _, _, filled_contour = extract_blue_contour(image_path)

contours, _ = cv2.findContours(filled_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# Wound measurements in pixels
wound_area_px = get_contour_area_px(contour)
wound_length_x_px, wound_length_y_px = get_contour_size_px(contour)

# Convert them to millimetres
wound_area_mm = get_contour_area_mm(contour, pixels_per_millimetre_ratio)
wound_length_x_mm, wound_length_y_mm = get_contour_size_mm(contour, pixels_per_millimetre_ratio)

print()
print(f'Wound Length X: {"{:.2f}px".format(wound_length_x_px)}, {"{:.2f}mm".format(wound_length_x_mm)}')
print(f'Wound Length Y: {"{:.2f}px".format(wound_length_y_px)}, {"{:.2f}mm".format(wound_length_y_mm)}')
print(f'Wound Area: {"{:.2f}px^2".format(wound_area_px)}, {"{:.2f}mm^2".format(wound_area_mm)}')

### VISUALISATIONS ###

# For this version of size_test, I want to see if I can calculate the padding and scaling amounts, similar to size_test2
# and use that instead to determine the position offset of the contour. I will then attempt to use cv2.resize() to scale up
# the contour, rather than trying to scale up the binary mask. My hope is that this will produce a clearer contour than size_test2
# without having to shrink the image down to a size that is difficult to see or work with like in size_test3

## Determine the padding and scaling factors

# Get the width and height of the original image
original_image_width, original_image_height = image.shape[:2]

# Make a copy of the original image, and then scale it down (without padding it) to determine
# how much padding was added to the image passed in to extract_blue_contour()
dummy_image = image.copy()

# Scale the image down to the same target size used in extract_blue_contour()
target_size = 256 # Copied from extract_blue_contour
dummy_image = resize_with_aspect_ratio(dummy_image, target_size)

## Pad the image as done in extract_blue_contour()
## TODO: Might not be needed. Cropping code blocked below
dummy_image = pad_image(dummy_image, target_size)

# Get the width and height of the dummy image
dummy_image_width, dummy_image_height = dummy_image.shape[:2]

# Calculate the scaling factors for each image axis
#scaling_factor_x = original_image_width / dummy_image_width
scaling_factor_x = int(original_image_width / dummy_image_width) + 1
#scaling_factor_y = original_image_height / dummy_image_height
scaling_factor_y = int(original_image_height / dummy_image_height) + 1

print('SF Float:', original_image_width / dummy_image_width)
print('SF Int:', scaling_factor_x)

# Calculate how much padding was added to the image in extract_blue_contour()
padding_amount_x = 0
padding_amount_y = 0

cropped_contour = contour.copy()

if dummy_image_width < target_size:
	# Padding was added to the X axis of the original image
	
	padding_amount_x = (target_size - dummy_image_width) // 2 # Halved because this amount is added equally to both sides of the image
	#cropped_contour = cropped_contour[padding_amount_x:-padding_amount_x, :]

if dummy_image_height < target_size:
	# Padding was added to the Y axis of the original image
	
	padding_amount_y = (target_size - dummy_image_height) // 2 # Halved because this amount is added equally to both sides of the image
	#cropped_contour = cropped_contour[padding_amount_x:-padding_amount_x, :]

print()
print('Contour Length: ', len(contour))
print('Contour Shape: ', contour.shape)
print('Scaling Factor X: ', scaling_factor_x)
print('Scaling Factor Y: ', scaling_factor_y)
print('Scaling Factor Type: ', type(scaling_factor_x))
print('Contour point type: ', type(contour[:, :, 0]))
print()

# Scale the contour up using the scaling factors
#scaled_contour = cv2.resize(contour, None, fx=scaling_factor_x, fy=scaling_factor_y)
scaled_contour = cropped_contour.copy()
scaled_contour[:, :, 0] *= scaling_factor_x
scaled_contour[:, :, 1] *= scaling_factor_y

## End padding and scaling factor calculation

# Coin Visualisations

image_circle_area = image.copy()
cv2.imshow('Wound Area', image_circle_area)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_circle_area = visualise_circle_area_mm(image_circle_area, best_circle, coin_area_mm)

image_circle_radius = image.copy()
image_circle_radius = visualise_circle_radius_mm(image_circle_radius, best_circle, coin_radius_mm)

image_circle_diameter = image.copy()
image_circle_diameter = visualise_circle_diameter_mm(image_circle_diameter, best_circle, coin_radius_mm * 2)

cv2.imshow('Coin Area', image_circle_area)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Coin Radius', image_circle_radius)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Coin Diameter', image_circle_diameter)
cv2.waitKey(0)
cv2.destroyAllWindows()

### END POINTLESS CODE ###

### TESTING

#epsilon = 0.0025 * cv2.arcLength(scaled_contour, True)
epsilon = 0.00197 * cv2.arcLength(scaled_contour, True)
scaled_contour = cv2.approxPolyDP(scaled_contour, epsilon, True)

### END TESTING

# Generate and display visualisation images
image_contour_area = image.copy()
image_contour_area = visualise_contour_area_mm(image_contour_area, scaled_contour, wound_area_mm)

### TESTING

for point in scaled_contour:
	x, y = point[0]
	cv2.circle(image_contour_area, (x, y), 3, (0, 0, 255), -1)

### END TESTING

cv2.imshow('Wound Area', image_contour_area)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_contour_size = image.copy()
image_contour_size = visualise_contour_size_mm(image_contour_size, scaled_contour, wound_length_x_mm, wound_length_y_mm)

cv2.imshow('Wound Size', image_contour_size)
cv2.waitKey(0)
cv2.destroyAllWindows()
