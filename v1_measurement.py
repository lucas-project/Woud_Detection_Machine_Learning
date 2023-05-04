import math
import numpy as np
import cv2

from v1_coin import detect_coin

### UTILITY FUNCTIONS ###

# Calculates the length in pixels between two points
def get_length_px(point_a, point_b):
	x1, y1 = point_a
	x2, y2 = point_b
	
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Returns the a point (X, Y) in the middle of two points
def get_midpoint(point_a, point_b):
	x1, y1 = point_a
	x2, y2 = point_b
	
	midpoint_x = (x1 + x2) // 2 # Uses integer division as this function deals with pixel coordinates (you can't half a pixel)
	midpoint_y = (y1 + y2) // 2
	
	return (midpoint_x, midpoint_y)

### CONVERSION FUNCTIONS ###

# Convert pixel measurements to millimeters
# NOTE: Set exponent=2 if converting area measurements!
def pixels_to_millimeters(pixels, pixels_per_millimeter_ratio, exponent=1):
	return pixels / (pixels_per_millimeter_ratio ** exponent)

# Convert millimeter measurements to pixels
# NOTE: Set exponent=2 if converting area measurements!
def millimeters_to_pixels(millimeters, pixels_per_millimeter_ratio, exponent=1):
	return millimeters * (pixels_per_millimeter_ratio ** exponent)

# TODO: Should this function also find the coin, or should that be separate?
# TODO: Write comments once this function is finalised

#def calculate_pixels_per_millimeter_ratio(image, reference_radius_mm):
#	best_circle, _ = detect_coin(image, min_radius=30, max_radius=100) # TODO: Refactor detect_coin?
#	_, _, radius_px = best_circle
#
#	pixels_per_millimeter = radius_px / reference_radius_mm
#	
#	return pixels_per_millimeter

# TODO: Refer to above comments.
# Calculate the pixels-per-millimeter ratio
# This function is just to avoid confusion as pixels_to_millimeters() could be used directly
def calculate_pixels_per_millimeter_ratio(circle_radius_px, reference_radius_mm):
	return pixels_to_millimeters(circle_radius_px, reference_radius_mm)

### CONTOUR MEASUREMENTS ###

# Get the area of a contour in pixels^2
def get_contour_area_px(contour):
	area_px = cv2.contourArea(contour)
	
	return area_px # TODO: Confirm return type

# Get the area of a contour in millimeters^2
# NOTE: Set exponent=2 if converting area measurements!
def get_contour_area_mm(contour, pixels_per_millimeter_ratio):
	# Get the area in pixels^2 first
	area_px = get_contour_area_px(contour)
	
	# Convert the result to millimeters^2
	area_mm = pixels_to_millimeters(area_px, pixels_per_millimeter_ratio, 2)
	
	return area_mm # TODO: Confirm return type

# Get the X and Y dimensions of a countour in pixels
def get_contour_size_px(contour):
	# Calculate a rect around the contour.
	rect = cv2.minAreaRect(contour)
	
	# Covert the rect to a list of points
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	
	# Get the four corner points of the box
	top_left, top_right, bottom_right, bottom_left = box
	
	## Calculate the midpoints for each edge
	#top_midpoint = get_midpoint(top_left, top_right)
	#bottom_midpoint = get_midpoint(bottom_left, bottom_right)
	#right_midpoint = get_midpoint(top_right, bottom_right)
	#left_midpoint = get_midpoint(top_left, bottom_left)
	
	## Calculate the lengths of the X and Y axes
	#x_length_px = get_length_px(top_midpoint, bottom_midpoint)
	#y_length_px = get_length_px(left_midpoint, right_midpoint)
	
	# TODO: I don't actually think the midpoints are required to measure lengths
	#       This code is leftover from the visualisation code, which will likely be done elsewhere
	
	# Calculate the pixel legnths of each axis
	x_length_px = get_length_px(top_left, top_right)
	y_length_px = get_length_px(top_left, bottom_left)
	
	# TODO: Should this function also display its result, or should that be separate?
	#boxed_image = image.copy()
	#cv2.drawContours(boxed_image, [box], 0, (0, 255, 0), 2)
	
	return x_length_px, y_length_px  # TODO: Confirm return type

# Get the X and Y dimensions of a countour in millimeters
def get_contour_size_mm(contour, pixels_per_millimeter_ratio):
	# Get the size in pixels first
	x_length_px, y_length_px = get_contour_size_px(contour)
	
	# Convert the result to millimeters
	x_length_mm = pixels_to_millimeters(x_length_px, pixels_per_millimeter_ratio)
	y_length_mm = pixels_to_millimeters(y_length_px, pixels_per_millimeter_ratio)
	
	return x_length_mm, y_length_mm # TODO: Confirm return type

### CIRCLE MEASUREMENTS ###

# Get the area of a circle in pixels^2
def get_circle_area_px(radius_px):
	return math.pi * (radius_px ** 2)

def get_circle_area_mm(coin_area_px, pixels_per_millimeter_ratio):
	return pixels_to_millimeters(coin_area_px, pixels_per_millimeter_ratio, 2)

def get_circle_radius_mm(radius_px, pixels_per_millimeter_ratio):
	return pixels_to_millimeters(radius_px, pixels_per_millimeter_ratio)

# Get the radius of a circle in pixels^2
def get_circle_radius_px(radius_mm, pixels_per_millimeter_ratio):
	return millimeters_to_pixels(radius_mm, pixels_per_millimeter_ratio)
