import unittest
import inspect
import numpy as np
import cv2

from v1_measurement import *
#from v1_coin import detect_coin
from v1_coin import *

# The margin of error allowance for a test
# This value represents a perctage of accuracy below 100%
MARGIN_OF_ERROR = 1

### UTILITY FUNCTIONS ###

# Used to calculate the accuracy of results
def calculate_accuracy(expected, result):
	accuracy = 100 - (abs(expected - result) / expected * 100)

	return round(accuracy, 2)

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

### TESTING CLASSES ###

class TestMeasurements(unittest.TestCase):
	def setUp(self):
		self.coin_radius_mm = 10.25
		
		self.image_path = 'contour/1.jpg'
		self.image = cv2.imread(self.image_path)

		best_circle, _ = detect_coin(self.image)
		self.coin_position_x, self.coin_position_y, self.coin_radius_px = best_circle
		
		self.pixels_per_millimetre_ratio = calculate_pixels_per_millimetre_ratio(self.coin_radius_px, self.coin_radius_mm)
		
		self.coin_area_px = get_circle_area_px(self.coin_radius_px)
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY)
	# 	calculate_pixels_per_millimetre_ratio
	# 	get_circle_radius_mm
	# 	pixels-to-millimetres
	def test_get_circle_radius_mm(self):
		expected = 10.25
		
		result = get_circle_radius_mm(self.coin_radius_px, self.pixels_per_millimetre_ratio)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Expected: {0}'.format(expected))
		print('Result:   {0}'.format(result))
		print('Accuracy: {0}%'.format(accuracy))
		
		self.assertEqual(expected, result, msg='Accuracy: {0}%'.format(accuracy))
		#self.assertGreaterEqual(accuracy, 100 - MARGIN_OF_ERROR, msg='Accuracy: {0}%'.format(accuracy))
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY)
	# 	calculate_pixels_per_millimetre_ratio
	# 	get_circle_area_px
	# 	pixels_to_millimetres
	def test_get_circle_area_mm(self):
		expected = 330.06
		
		result = get_circle_area_mm(self.coin_area_px, self.pixels_per_millimetre_ratio)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Expected: {0}'.format(expected))
		print('Result:   {0}'.format(result))
		print('Accuracy: {0}%'.format(accuracy))
		
		self.assertEqual(expected, result, msg='Accuracy: {0}%'.format(accuracy))
		#self.assertGreaterEqual(accuracy, 100 - MARGIN_OF_ERROR, msg='Accuracy: {0}%'.format(accuracy))
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY. Positions are used, but are not validated)
	# 	calculate_pixels_per_millimetre_ratio
	# 	get_contour_area_mm
	# 	get_contour_area_px
	# 	pixels_to_millimetres
	def test_get_contour_area_mm(self):
		expected = 330.06
		
		# Convert the circle to a contour
		contour = circle_to_contour(self.image, self.coin_position_x, self.coin_position_y, self.coin_radius_px)
		
		result = get_contour_area_mm(contour, self.pixels_per_millimetre_ratio)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Expected: {0}'.format(expected))
		print('Result:   {0}'.format(result))
		print('Accuracy: {0}%'.format(accuracy))
		
		self.assertGreaterEqual(accuracy, 100 - MARGIN_OF_ERROR, msg='Accuracy: {0}%'.format(accuracy))
	
	# Tests the following functions:
	# 	detect_coin (RADIUS ONLY. Positions are used, but are not validated)
	# 	calculate_pixels_per_millimetre_ratio
	# 	get_contour_size_mm
	# 	get_contour_size_px
	# 	pixels_to_millimetres
	def test_get_contour_size_mm(self):
		expected = (20.50, 20.50)
		
		# Convert the circle to a contour
		contour = circle_to_contour(self.image, self.coin_position_x, self.coin_position_y, self.coin_radius_px)
		
		x_length_mm, y_length_mm = get_contour_size_mm(contour, self.pixels_per_millimetre_ratio)
		
		# Calculate the accuracy of the results
		accuracy_x = calculate_accuracy(expected[0], x_length_mm)
		accuracy_y = calculate_accuracy(expected[1], y_length_mm)
		accuracy = (accuracy_x + accuracy_y) / 2
		
		# Round results to 2 decimal places
		result = (round(x_length_mm, 2), round(y_length_mm, 2))
		
		# Log the expected, result and accuracy
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Expected: {0}'.format(expected))
		print('Result:   {0}'.format(result))
		print('Accuracy: {0}%'.format(accuracy))
		
		#self.assertEqual(expected, result, msg='Accuracy: ({0}%, {1}%)'.format(accuracy_x, accuracy_y))
		self.assertGreaterEqual(accuracy, 100 - MARGIN_OF_ERROR, msg='Accuracy: {0}%'.format(accuracy))
	
	def tearDown(self):
		del self.coin_radius_mm
		del self.image
		del self.image_path
		del self.coin_position_x
		del self.coin_position_y
		del self.coin_radius_px
		del self.pixels_per_millimetre_ratio
		del self.coin_area_px

class TestCoin(unittest.TestCase):
	def setUp(self):
		self.coin_diameter_mm = 20.50
		
		self.image_path = 'contour/1.jpg'
		self.image = cv2.imread(self.image_path)

		best_circle, self.ratio_coin = detect_coin(self.image)
		self.coin_position_x, self.coin_position_y, self.coin_radius_px = best_circle
	
	# Tests the following functions:
	# 	NONE
	@unittest.skip("v1_coin.py does not contain a function that returns the radius in millimetres, or one which can convert it from pixels to millimetres")
	def test_coin_actual_radius(self):
		expected = 10.25
		
		result = None
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Expected: {0}'.format(expected))
		print('Result:   {0}'.format(result))
		print('Accuracy: {0}%'.format(accuracy))
		
		#self.assertEqual(expected, result, msg='Accuracy: {0}%'.format(accuracy))
		self.assertGreaterEqual(accuracy, 100 - MARGIN_OF_ERROR, msg='Accuracy: {0}%'.format(accuracy))
	
	# Tests the following functions:
	# 	v1_coin.py: detect_coin (RADIUS ONLY)
	# 	v1_coin.py: coin_actual_area
	def test_coin_actual_area(self):
		expected = 330.06
		
		result = calculate_actual_coin_area(self.coin_diameter_mm)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Expected: {0}'.format(expected))
		print('Result:   {0}'.format(result))
		print('Accuracy: {0}%'.format(accuracy))
		
		#self.assertEqual(expected, result, msg='Accuracy: {0}%'.format(accuracy))
		self.assertGreaterEqual(accuracy, 100 - MARGIN_OF_ERROR, msg='Accuracy: {0}%'.format(accuracy))
	
	# Tests the following functions:
	# 	v1_coin.py: detect_coin (RADIUS ONLY)
	# 	v1_coin.py: coin_actual_area
	# 	v1_coin.py: calculate_actual_wound_area
	def test_calculate_actual_wound_area(self):
		expected = 330.06
		
		# Convert the circle to a contour
		contour = circle_to_contour(self.image, self.coin_position_x, self.coin_position_y, self.coin_radius_px)
		
		# Extracted some data from extract_blue_contour to aproximate wound_ratio in the same manner as used in v1_coin.py
		original_height, original_width = self.image.shape[:2]
		
		if original_height > original_width:
			long_side = original_height
			short_side = original_width
		else:
			long_side = original_width
			short_side = original_height
		
		pixel_count = np.count_nonzero(contour)
		resize_ratio = 256 / long_side 
		resize_short_size = resize_ratio * short_side
		total_pixels = 256 * resize_short_size
		pixel_ratio = pixel_count / total_pixels
		# End approximation code
		
		ratio_wound = pixel_ratio
		coin_actual_area = calculate_actual_coin_area(self.coin_diameter_mm)
		
		result = calculate_actual_wound_area(self.ratio_coin, coin_actual_area, ratio_wound)
		
		# Calculate the accuracy of the result
		accuracy = calculate_accuracy(expected, result)
		
		# Round results to 2 decimal places
		result = round(result, 2)
		
		# Log the expected, result and accuracy
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Expected: {0}'.format(expected))
		print('Result:   {0}'.format(result))
		print('Accuracy: {0}%'.format(accuracy))
		
		#self.assertEqual(expected, result, msg='Accuracy: {0}%'.format(accuracy))
		self.assertGreaterEqual(accuracy, 100 - MARGIN_OF_ERROR, msg='Accuracy: {0}%'.format(accuracy))
	
	# Tests the following functions:
	# 	NONE
	@unittest.skip("v1_coin.py does not contain a function that returns the contour size. Other than some example code I wrote which is old, untested, and should be deleted")
	def test_estimate_actual_size(self):
		pass
	
	def tearDown(self):
		del self.coin_diameter_mm
		del self.image_path
		del self.image
		del self.ratio_coin
		del self.coin_position_x
		del self.coin_position_y
		del self.coin_radius_px

class TestComparisons(unittest.TestCase):
	def setUp(self):
		self.coin_radius_mm = 10.25
		self.coin_diameter_mm = 20.50
		
		self.image_path = 'contour/1.jpg'
		self.image = cv2.imread(self.image_path)

		best_circle, self.ratio_coin = detect_coin(self.image)
		self.coin_position_x, self.coin_position_y, self.coin_radius_px = best_circle
		
		self.pixels_per_millimetre_ratio = calculate_pixels_per_millimetre_ratio(self.coin_radius_px, self.coin_radius_mm)
		
		self.coin_area_px = get_circle_area_px(self.coin_radius_px)
	
	# Compares circle area detection functions from both v1_measurement.py and v1_coin.py
	def test_circle_area_comparison(self):
		expected = 330.06
		
		measurement_function = get_circle_area_mm
		coin_function = calculate_actual_coin_area
		
		measurement_result = measurement_function(self.coin_area_px, self.pixels_per_millimetre_ratio)
		coin_result = coin_function(self.coin_diameter_mm)
		
		# Calculate the accuracy of the results
		measurement_accuracy = calculate_accuracy(expected, measurement_result)
		coin_accuracy = calculate_accuracy(expected, coin_result)
		
		# Round results to 2 decimal places
		measurement_result = round(measurement_result, 2)
		coin_result = round(coin_result, 2)
		
		# Determine the winner and loser
		winner = '{0} (Accuracy: {1}%)'.format(measurement_function.__name__, measurement_accuracy)
		loser  = '{0} (Accuracy: {1}%)'.format(coin_function.__name__, coin_accuracy)
		
		if coin_accuracy > measurement_accuracy:
			winner = '{0} (Accuracy: {1}%)'.format(coin_function.__name__, coin_accuracy)
			loser  = '{0} (Accuracy: {1}%)'.format(measurement_function.__name__, measurement_accuracy)
		elif coin_accuracy == measurement_accuracy:
			winner = 'DRAW (Accuracy: {0}% - {1}%)'.format(measurement_accuracy, coin_accuracy)
			loser = '-'
		
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Winner:   {0}'.format(winner))
		print('Loser:    {0}'.format(loser))
		
		self.assertGreaterEqual(measurement_accuracy, coin_accuracy)
	
	# Compares contour area detection functions from both v1_measurement.py and v1_coin.py	
	def test_contour_area_comparison(self):
		expected = 330.06
		
		# Define the functions to test
		measurement_function = get_contour_area_mm
		coin_function = calculate_actual_wound_area
		
		# Convert the circle to a contour
		contour = circle_to_contour(self.image, self.coin_position_x, self.coin_position_y, self.coin_radius_px)
		
		# Extracted some data from extract_blue_contour to aproximate wound_ratio in the same manner as used in v1_coin.py
		original_height, original_width = self.image.shape[:2]
		
		if original_height > original_width:
			long_side = original_height
			short_side = original_width
		else:
			long_side = original_width
			short_side = original_height
		
		pixel_count = np.count_nonzero(contour)
		resize_ratio = 256 / long_side 
		resize_short_size = resize_ratio * short_side
		total_pixels = 256 * resize_short_size
		pixel_ratio = pixel_count / total_pixels
		# End approximation code
		
		ratio_wound = pixel_ratio
		coin_actual_area = calculate_actual_coin_area(self.coin_diameter_mm)
		
		# Calculate results for both functions
		measurement_result = measurement_function(contour, self.pixels_per_millimetre_ratio)
		coin_result = coin_function(self.ratio_coin, coin_actual_area, ratio_wound)
		
		# Calculate the accuracy of the results
		measurement_accuracy = calculate_accuracy(expected, measurement_result)
		coin_accuracy = calculate_accuracy(expected, coin_result)
		
		# Round results to 2 decimal places
		measurement_result = round(measurement_result, 2)
		coin_result = round(coin_result, 2)
		
		# Determine the winner and loser
		winner = '{0} (Accuracy: {1}%)'.format(measurement_function.__name__, measurement_accuracy)
		loser  = '{0} (Accuracy: {1}%)'.format(coin_function.__name__, coin_accuracy)
		
		if coin_accuracy > measurement_accuracy:
			winner = '{0} (Accuracy: {1}%)'.format(coin_function.__name__, coin_accuracy)
			loser  = '{0} (Accuracy: {1}%)'.format(measurement_function.__name__, measurement_accuracy)
		elif coin_accuracy == measurement_accuracy:
			winner = 'DRAW (Accuracy: {0}% - {1}%)'.format(measurement_accuracy, coin_accuracy)
			loser = '-'
		
		print()
		print('Function: {0}'.format(inspect.currentframe().f_code.co_name))
		print('Winner:   {0}'.format(winner))
		print('Loser:    {0}'.format(loser))
		
		self.assertGreaterEqual(measurement_accuracy, coin_accuracy)
	
	def tearDown(self):
		del self.coin_radius_mm
		del self.coin_diameter_mm
		del self.image
		del self.image_path
		del self.coin_position_x
		del self.coin_position_y
		del self.coin_radius_px
		del self.pixels_per_millimetre_ratio
		del self.coin_area_px
		del self.ratio_coin

### RUN THE TESTS ###

# measurement.py tests

print()
print('UNIT TESTING: v1_measurement.py')

measurement_tests = unittest.TestLoader().loadTestsFromTestCase(TestMeasurements)

runner = unittest.TextTestRunner()
runner.run(measurement_tests)

print('----------------------------------------------------------------------')

# v1_coin.py tests

print()
print('UNIT TESTING: v1_coin.py')

coin_tests = unittest.TestLoader().loadTestsFromTestCase(TestCoin)

runner = unittest.TextTestRunner()
runner.run(coin_tests)

print('----------------------------------------------------------------------')

print()
print('COMPARING TEST RESULTS')

comparison_tests = unittest.TestLoader().loadTestsFromTestCase(TestComparisons)

runner = unittest.TextTestRunner()
runner.run(comparison_tests)
