import math
import numpy as np
import cv2

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

# Convert pixel measurements to millimetres
# NOTE: Set exponent=2 if converting area measurements!
def pixels_to_millimetres(pixels, pixels_per_millimetre_ratio, exponent=1):
	return pixels / (pixels_per_millimetre_ratio ** exponent)

# Convert millimetre measurements to pixels
# NOTE: Set exponent=2 if converting area measurements!
def millimetres_to_pixels(millimetres, pixels_per_millimetre_ratio, exponent=1):
	return millimetres * (pixels_per_millimetre_ratio ** exponent)

# Calculate the pixels-per-millimetre ratio
# This function is just to avoid confusion as pixels_to_millimetres() could be used directly
def calculate_pixels_per_millimetre_ratio(circle_radius_px, reference_radius_mm):
	return pixels_to_millimetres(circle_radius_px, reference_radius_mm)

### CONTOUR MEASUREMENTS ###

# Get the area of a contour in pixels^2
def get_contour_area_px(contour):
	area_px = cv2.contourArea(contour)
	
	return area_px

# Get the area of a contour in millimetres^2
# NOTE: Set exponent=2 if converting area measurements!
def get_contour_area_mm(contour, pixels_per_millimetre_ratio):
	# Get the area in pixels^2 first
	area_px = get_contour_area_px(contour)
	
	# Convert the result to millimetres^2
	area_mm = pixels_to_millimetres(area_px, pixels_per_millimetre_ratio, 2)
	
	return area_mm

# Get the X and Y dimensions of a countour in pixels
def get_contour_size_px(contour):
	# Calculate a rect around the contour.
	rect = cv2.minAreaRect(contour)
	
	# Covert the rect to a list of points
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	
	# Get the four corner points of the box
	top_left, top_right, bottom_right, bottom_left = box
	
	# Calculate the pixel legnths of each axis
	x_length_px = get_length_px(top_left, top_right)
	y_length_px = get_length_px(top_left, bottom_left)
	
	return x_length_px, y_length_px

# Get the X and Y dimensions of a countour in millimetres
def get_contour_size_mm(contour, pixels_per_millimetre_ratio):
	# Get the size in pixels first
	x_length_px, y_length_px = get_contour_size_px(contour)
	
	# Convert the result to millimetres
	x_length_mm = pixels_to_millimetres(x_length_px, pixels_per_millimetre_ratio)
	y_length_mm = pixels_to_millimetres(y_length_px, pixels_per_millimetre_ratio)
	
	return x_length_mm, y_length_mm

### CIRCLE MEASUREMENTS ###

# Get the area of a circle in pixels^2
def get_circle_area_px(radius_px):
	return math.pi * (radius_px ** 2)

def get_circle_area_mm(coin_area_px, pixels_per_millimetre_ratio):
	return pixels_to_millimetres(coin_area_px, pixels_per_millimetre_ratio, 2)

def get_circle_radius_mm(radius_px, pixels_per_millimetre_ratio):
	return pixels_to_millimetres(radius_px, pixels_per_millimetre_ratio)

# Get the radius of a circle in pixels^2
def get_circle_radius_px(radius_mm, pixels_per_millimetre_ratio):
	return millimetres_to_pixels(radius_mm, pixels_per_millimetre_ratio)
