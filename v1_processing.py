# I was asked not to overwrite get_blue_contours() but to instead
# asked to make a new function. I don't know where else to put it
# so I made a new file just to store it. Not ideal.

import numpy as np
import cv2

# Extracts a contour from an image which has a blue outline drawn on it
def extract_contours_from_outlined_image(image):
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	# Define the lower and upper boundaries for the blue color
	lower_blue = np.array([100, 50, 50])
	upper_blue = np.array([130, 255, 255])
	
	# Create a mask that isolates the blue color in the image
	blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
	
	# Perform contour detection
	contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	## Select the largest contour from the list
	#largest_contour = max(contours, key=cv2.contourArea)
	
	# Return the largest contour
	#return largest_contour
	
	# Return list of contours
	return contours