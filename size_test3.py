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

# For this version of size_test, I want to test the visualisations on the scaled down image passed in to get_blue_contour
# This image scale is also what is used for training the model, so this could be a valuable approach
# However, because the rescaling is done INSIDE get_blue_contour for some reason, we need to repeat it here

## Repeat the scaling techinque applied to the image

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

## End scaling repeat

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

# Generate and display visualisation images
image_contour_area = image.copy()
image_contour_area = visualise_contour_area_mm(image_contour_area, contour, wound_area_mm)

cv2.imshow('Wound Area', image_contour_area)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_contour_size = image.copy()
image_contour_size = visualise_contour_size_mm(image_contour_size, contour, wound_length_x_mm, wound_length_y_mm)

cv2.imshow('Wound Size', image_contour_size)
cv2.waitKey(0)
cv2.destroyAllWindows()
