import math
import numpy as np
import cv2
import tensorflow as tf

from keras.models import load_model
from v1_coin import detect_coin
from v1_border import extract_blue_contour, resize_with_aspect_ratio, pad_image
from v1_evaluation import load_evaluation_images

#input_directory = 'contour/'
image_path = 'contour/1.jpg'

image = cv2.imread(image_path)

_, _, _, filled_contour = extract_blue_contour(image_path)

## Need to pad the original image, as it's being done in extract_blue_contour() to generate the binary mmask, but the original is still too big!
#target_size = 256
#image = resize_with_aspect_ratio(image, target_size)
#image = pad_image(image, target_size)

# NOTE: The above resizing code makes the coins from some images impossible (e.g. contour/1.jpg)

contours, _ = cv2.findContours(filled_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# Width/Height
rect = cv2.minAreaRect(contour)
#print (len(contours))
#print (contour.shape)

box = cv2.boxPoints(rect)
box = np.int0(box)

boxed_image = image.copy()

# Need to pad the original image, as it's being done in extract_blue_contour() to generate the binary mmask, but the original is still too big!
target_size = 256
boxed_image = resize_with_aspect_ratio(boxed_image, target_size)
boxed_image = pad_image(boxed_image, target_size)

cv2.drawContours(boxed_image, [box], 0, (0, 255, 0), 2)

# Area
area_px = cv2.contourArea(contour)

####

# PIXELS PER METRIC RATIO

# The radius of an Australian $2 coin in millimetres
coin_radius_mm = 10.25
#coin_radius_mm = 6.5 # FOR VALIDATION ONLY! This is the size of the coin on screen, if a ruler is held up to the screen.

# Get the radius of the found coin in pixels
best_circle, _ = detect_coin(image)
coin_x, coin_y, radius_px = best_circle

pixels_per_metric = radius_px / coin_radius_mm

###

#image_fancy = image.copy()
image_fancy = boxed_image.copy()

# Draw a ring around the coin
cv2.circle(image_fancy, (coin_x, coin_y), radius_px, (0, 255, 0), 2)

# Draw a circle at the midpoint
cv2.circle(image_fancy, (coin_x, coin_y), 5, (0, 0, 255), -1)

# Draw a circle a the edge
cv2.circle(image_fancy, (coin_x + radius_px, coin_y), 5, (0, 0, 255), -1)

# Draw a line from the center to the edge
cv2.line(image_fancy, (coin_x, coin_y), (coin_x + radius_px, coin_y), (255, 0, 255), 2)

# Draw circle at corners

for x, y in box:
    cv2.circle(image_fancy, (int(x), int(y)), 5, (255, 0, 0), -1)

# Draw circle at midpoints between cornerse

top_left, top_right, bottom_right, bottom_left = box

top_midpoint = (top_left + top_right) // 2
bottom_midpoint = (bottom_left + bottom_right) // 2
right_midpoint = (top_right + bottom_right) // 2
left_midpoint = (top_left + bottom_left) // 2

top_midpoint_x, top_midpoint_y = (top_left + top_right) // 2
bottom_midpoint_x, bottom_midpoint_y = (bottom_left + bottom_right) // 2
left_midpoint_x, left_midpoint_y = (top_left + bottom_left) // 2
right_midpoint_x, right_midpoint_y = (top_right + bottom_right) // 2

cv2.circle(image_fancy, (top_midpoint_x, top_midpoint_y), 5, (0, 0, 255), -1)
cv2.circle(image_fancy, (bottom_midpoint_x, bottom_midpoint_y), 5, (0, 0, 255), -1)
cv2.circle(image_fancy, (left_midpoint_x, left_midpoint_y), 5, (0, 0, 255), -1)
cv2.circle(image_fancy, (right_midpoint_x, right_midpoint_y), 5, (0, 0, 255), -1)

# Draw lines between midpoint circles

cv2.line(image_fancy, top_midpoint, bottom_midpoint, (255, 0, 255), 2)
cv2.line(image_fancy, left_midpoint, right_midpoint, (255, 0, 255), 2)

####

# REAL WORLD MEASUREMENTS

# Calculate the lengths of the X and Y axes
x_length_px = math.sqrt((top_midpoint_x - bottom_midpoint_x)**2 + (top_midpoint_y - bottom_midpoint_y)**2)
y_length_px = math.sqrt((left_midpoint_x - right_midpoint_x)**2 + (left_midpoint_y - right_midpoint_y)**2)
#print(f'X: {x_length_px} Y: {y_length_px}')

# Convert these to millimetres
x_length_mm = x_length_px / pixels_per_metric
y_length_mm = y_length_px / pixels_per_metric
#print(f'X: {x_length_mm} Y: {y_length_mm}')

# Draw these on the image
cv2.putText(image_fancy, "{:.2f}mm".format(x_length_mm), (int(top_midpoint_x - 15), int(top_midpoint_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
cv2.putText(image_fancy, "{:.2f}mm".format(y_length_mm), (int(left_midpoint_x - 15), int(left_midpoint_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

# Repeat the above for the coin. Might not be needed!
coin_radius_mm = radius_px / pixels_per_metric
cv2.putText(image_fancy, "{:.2f}mm".format(coin_radius_mm), (int(coin_x + radius_px - 15), int(coin_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

# Caclulate Wound area
area_mm = area_px / pixels_per_metric ** 2

# Draw area on the image
#cv2.putText()

# Do the same for the coin. Might not be needed!
coin_area_px = math.pi * (radius_px ** 2)
coin_area_mm = math.pi * (coin_radius_mm ** 2)

####

print(f'Pixels per Metric Ratio: {pixels_per_metric}')
print()
print(f'Coin Radius: {"{:.2f}px".format(radius_px)}, {"{:.2f}mm".format(coin_radius_mm)}')
print(f'Coin Area: {"{:.2f}px^2".format(coin_area_px)}, {"{:.2f}mm^2".format(coin_area_mm)}')
print()
print(f'Wound Length X: {"{:.2f}px".format(x_length_px)}, {"{:.2f}mm".format(x_length_mm)}')
print(f'Wound Length Y: {"{:.2f}px".format(y_length_px)}, {"{:.2f}mm".format(y_length_mm)}')
print(f'Wound Area: {"{:.2f}px^2".format(area_px)}, {"{:.2f}mm^2".format(area_mm)}')

cv2.imshow('Fancy Image', image_fancy)
cv2.waitKey(0)
cv2.destroyAllWindows()

quit()

###

def dice_coefficient(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

with tf.keras.utils.custom_object_scope({'dice_coefficient': dice_coefficient}):
    model = load_model('wound_segmentation_model.h5')

evaluation_images = load_evaluation_images('fake_evaluation/', True)

predicted_masks = model.predict(evaluation_images)

threshold = 0.6

binary_masks = (predicted_masks > threshold).astype(np.uint8) * 255

for i, mask in enumerate(binary_masks):
	contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print(f'Contours: {len(contours)}')
	image = cv2.imread(f'fake_evaluation/{i}.jpg') # Hack code. Fix this later
	for contour in contours:
		rect = cv2.minAreaRect(contour)
		
		box = cv2.boxPoints(rect)
		box = box.astype(int)
		
		cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
		cv2.imshow('Boxed Image', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

print('Done')
print(f'Predicted Masks: {len(predicted_masks)}')
