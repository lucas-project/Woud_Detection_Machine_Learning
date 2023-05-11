import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from v1_measurement import get_midpoint

font_path = 'fonts/OpenSans-Regular.ttf'
font_size = 24
font = ImageFont.truetype(font_path, font_size)

### HELPER FUNCTIONS ###

def drawText(image, text, position_x, position_y):
	image_pil = Image.fromarray(image)
	draw = ImageDraw.Draw(image_pil)
	draw.text((int(position_x), int(position_y)), text, font=font, fill=(255, 255, 255))
	image = np.array(image_pil)
	
	return image

### VISUALISATION FUNCTIONS ###

# Visualise a OpenCV circle showing its area in millimetres
def visualise_circle_area_mm(image, circle, area_mm):
	position_x, position_y, radius_px = circle
	
	# Create a copy of the image
	overlay_image = image.copy()
	
	# Draw a cirle on the overlay image
	cv2.circle(overlay_image, (position_x, position_y), radius_px, (255, 0, 0), -1)
	
	# Overlay the image on the original using alpha transparency
	alpha = 0.5
	image = cv2.addWeighted(image, 1 - alpha, overlay_image, alpha, 0)
	
	# Draw a ring around the circle
	cv2.circle(image, (position_x, position_y), radius_px, (255, 0, 0), 2)
	
	# Add the area measurement as text
	image = drawText(image, "{:.2f}mm²".format(area_mm), position_x, position_y)
	
	return image

# Visualise a OpenCV circle showing its radius in millimetres
def visualise_circle_radius_mm(image, circle, radius_mm):
	position_x, position_y, radius_px = circle
	
	# Draw a ring around the circle
	cv2.circle(image, (position_x, position_y), radius_px, (0, 255, 0), 2)
	
	# Draw a dot at the midpoint
	cv2.circle(image, (position_x, position_y), 5, (0, 0, 255), -1)
	
	# Draw a dot a the edge
	cv2.circle(image, (position_x + radius_px, position_y), 5, (0, 0, 255), -1)
	
	# Draw a line from the center to the edge
	cv2.line(image, (position_x, position_y), (position_x + radius_px, position_y), (255, 0, 255), 2)
	
	# Add the radius measurement as text
	image = drawText(image, "{:.2f}mm".format(radius_mm), position_x + radius_px - 15, position_y - 10)
	
	return image

# Visualise a OpenCV circle showing its diameter in millimetres
def visualise_circle_diameter_mm(image, circle, diameter_mm):
	position_x, position_y, radius_px = circle
	
	# Draw a ring around the circle
	cv2.circle(image, (position_x, position_y), radius_px, (0, 255, 0), 2)
	
	# Draw a dot at each edge
	cv2.circle(image, (position_x - radius_px, position_y), 5, (0, 0, 255), -1)
	cv2.circle(image, (position_x + radius_px, position_y), 5, (0, 0, 255), -1)
	
	# Draw a line through the center of the circle
	cv2.line(image, (position_x - radius_px, position_y), (position_x + radius_px, position_y), (255, 0, 255), 2)
	
	# Add the diameter measurement as text
	image = drawText(image, "{:.2f}mm".format(diameter_mm), position_x + radius_px - 15, position_y - 10)
	
	return image

# Visualise an OpenCV contour showing its area in millimetres
def visualise_contour_area_mm(image, contour, area_mm):
	# Create a copy of the image
	overlay_image = image.copy()
	
	# Draw the contour on the overlay image
	cv2.drawContours(overlay_image, [contour], -1, (255, 0, 0), -1)
	
	# Overlay the image on the original using alpha transparency
	alpha = 0.5
	image = cv2.addWeighted(image, 1 - alpha, overlay_image, alpha, 0)
	
	# Draw the outline of the contour
	cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
	
	# Get the centre point of the contour
	moments = cv2.moments(contour)
	center_x = int(moments["m10"] / moments["m00"])
	center_y = int(moments["m01"] / moments["m00"])
	
	# Add the area measurement as text
	image = drawText(image, "{:.2f}mm²".format(area_mm), center_x, center_y)
	
	return image

# Visualise an OpenCV contour showing width and height in millimetres
def visualise_contour_size_mm(image, contour, size_x_mm, size_y_mm):
	# Calculate a rect around the contour.
	rect = cv2.minAreaRect(contour)
	
	# Covert the rect to a list of points
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	
	# Draw the box around the contour
	cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
	
	# Get the corners of the box
	top_left, top_right, bottom_right, bottom_left = box
	
	# Draw a circle at each corner
	cv2.circle(image, top_left, 5, (255, 0, 0), -1)
	cv2.circle(image, top_right, 5, (255, 0, 0), -1)
	cv2.circle(image, bottom_right, 5, (255, 0, 0), -1)
	cv2.circle(image, bottom_left, 5, (255, 0, 0), -1)
	
	# Calculate the midpoints for each edge
	top_midpoint = get_midpoint(top_left, top_right)
	bottom_midpoint = get_midpoint(bottom_left, bottom_right)
	left_midpoint = get_midpoint(top_left, bottom_left)
	right_midpoint = get_midpoint(top_right, bottom_right)
	
	# Draw circles at middle of each edge
	cv2.circle(image, top_midpoint, 5, (0, 0, 255), -1)
	cv2.circle(image, bottom_midpoint, 5, (0, 0, 255), -1)
	cv2.circle(image, left_midpoint, 5, (0, 0, 255), -1)
	cv2.circle(image, right_midpoint, 5, (0, 0, 255), -1)
	
	# Draw lines between midpoint circles
	cv2.line(image, top_midpoint, bottom_midpoint, (255, 0, 255), 2)
	cv2.line(image, left_midpoint, right_midpoint, (255, 0, 255), 2)
	
	# Add the size measurements as text
	image = drawText(image, "{:.2f}mm".format(size_x_mm), top_midpoint[0] - 15, top_midpoint[1] - 10)
	image = drawText(image, "{:.2f}mm".format(size_y_mm), left_midpoint[0] - 15, left_midpoint[1] - 10)
	
	return image
