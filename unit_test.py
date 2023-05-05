import unittest
import logging
import inspect
import numpy as np
import cv2

from v1_measurement import *
from v1_coin import detect_coin

### HELPER FUNCTIONS ###

# Used to calculate the accuracy of results
# Returned as a string, e.g.: "[Accuracy]%"
def calculate_accuracy(expected, result):
	accuracy = 100 - ((expected - result) / expected * 100)

	return '{0}%'.format(round(accuracy, 2))

# Convert a circle to a contour
# This is used to test the wound measurements (contours)
# using the circle dectected for the coin as it's the
# only object we know the real-world sizes for.
def circle_to_contour(image, position_x, position_y, radius_px):
	result = np.zeros_like(image)
	cv2.circle(result, (position_x, position_y), radius_px, (255, 255, 255), thickness=-1)
	
	result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	result = result.astype(np.uint8)
	
	contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	result = max(contours, key=cv2.contourArea)
	
	return result

# This function is just a tidy place to initialise required values
def init_values(coin_radius_mm):
	image_path = 'contour/1.jpg'

	image = cv2.imread(image_path)

	best_circle, _ = detect_coin(image)
	coin_position_x, coin_position_y, coin_radius_px = best_circle
	
	pixels_per_millimeter_ratio = calculate_pixels_per_millimeter_ratio(coin_radius_px, coin_radius_mm)
	
	coin_area_px = get_circle_area_px(coin_radius_px)
	
	return image, coin_position_x, coin_position_y, coin_radius_px, pixels_per_millimeter_ratio, coin_area_px

### TESTING CLASSES ###

class TestMeasurements(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		logging.basicConfig(level=logging.INFO)
	
	def setUp(self):
		self.coin_radius_mm = 10.25
		self.image, self.coin_position_x, self.coin_position_y, self.coin_radius_px, self.pixels_per_millimeter_ratio, self.coin_area_px = init_values(self.coin_radius_mm)
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY)
	# 	calculate_pixels_per_millimeter_ratio
	# 	get_circle_radius_mm
	# 	pixels-to-millimeters
	def test_get_circle_radius_mm(self):
		expected = 10.25
		
		result = get_circle_radius_mm(self.coin_radius_px, self.pixels_per_millimeter_ratio)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		logging.info('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		logging.info('Expected: {0}'.format(expected))
		logging.info('Result:   {0}'.format(result))
		logging.info('Accuracy: {0}'.format(accuracy))
		
		self.assertEqual(expected, result, msg='Accuracy: {0}'.format(accuracy))
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY)
	# 	calculate_pixels_per_millimeter_ratio
	# 	get_circle_area_px
	# 	pixels_to_millimeters
	def test_get_circle_area_mm(self):
		expected = 330.06
		
		result = get_circle_area_mm(self.coin_area_px, self.pixels_per_millimeter_ratio)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		logging.info('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		logging.info('Expected: {0}'.format(expected))
		logging.info('Result:   {0}'.format(result))
		logging.info('Accuracy: {0}'.format(accuracy))
		
		self.assertEqual(expected, result, msg='Accuracy: {0}'.format(accuracy))
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY. Positions are used, but are not validated)
	# 	calculate_pixels_per_millimeter_ratio
	# 	get_contour_area_mm
	# 	get_contour_area_px
	# 	pixels_to_millimeters
	def test_get_contour_area_mm(self):
		expected = 330.06
		
		# Convert the circle to a contour
		contour = circle_to_contour(self.image, self.coin_position_x, self.coin_position_y, self.coin_radius_px)
		
		result = get_contour_area_mm(contour, self.pixels_per_millimeter_ratio)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		logging.info('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		logging.info('Expected: {0}'.format(expected))
		logging.info('Result:   {0}'.format(result))
		logging.info('Accuracy: {0}'.format(accuracy))
		
		self.assertEqual(expected, result, msg='Accuracy: {0}'.format(accuracy))
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY. Positions are used, but are not validated)
	# 	calculate_pixels_per_millimeter_ratio
	# 	get_contour_size_mm
	# 	get_contour_size_px
	# 	pixels_to_millimeters
	def test_get_contour_size_mm(self):
		expected = (20.50, 20.50)
		
		# Convert the circle to a contour
		contour = circle_to_contour(self.image, self.coin_position_x, self.coin_position_y, self.coin_radius_px)
		
		x_length_mm, y_length_mm = get_contour_size_mm(contour, self.pixels_per_millimeter_ratio)
		
		# Calculate the accuracy of the results
		accuracy_x = calculate_accuracy(expected[0], x_length_mm)
		accuracy_y = calculate_accuracy(expected[1], y_length_mm)
		accuracy = (accuracy_x, accuracy_y)
		
		# Round results to 2 decimal places
		result = (round(x_length_mm, 2), round(y_length_mm, 2))
		
		# Log the expected, result and accuracy
		print()
		logging.info('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		logging.info('Expected: {0}'.format(expected))
		logging.info('Result:   {0}'.format(result))
		logging.info('Accuracy: {0}'.format(accuracy))
		
		self.assertEqual(expected, result, msg='Accuracy: {0}'.format(accuracy))
	
	def tearDown(self):
		del self.coin_radius_mm
		del self.image
		del self.coin_position_x
		del self.coin_position_y
		del self.coin_radius_px
		del self.pixels_per_millimeter_ratio
		del self.coin_area_px

### RUN THE TESTS ###

unittest.main()
