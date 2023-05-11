import os
import cv2

from datetime import datetime
from v1_coin import detect_coin
from v1_measurement import calculate_pixels_per_millimetre_ratio, get_circle_area_px, get_circle_area_mm, get_contour_size_px, get_contour_area_px, get_contour_size_mm, get_contour_area_mm
from v1_processing import extract_contour_from_outlined_image
from v1_visualisation import visualise_circle_area_mm, visualise_circle_radius_mm, visualise_circle_diameter_mm, visualise_contour_area_mm, visualise_contour_size_mm

# TODO: Add model paths
# TODO: Confirm model file name

model_file_name = 'wound_segmentation_model.h5'

COIN_RADIUS_MM = 10.25 # The radius of an Australian $2 coin in millimetres

### PROGRAM TASK FUNCTIONS ###

def task_wound_analysis_image_processing():
	while(True):
		print('------------------------------------------')
		print('              WOUND ANALYSIS              ')
		print('             Image Processing             ')
		print('------------------------------------------')
		
		# The path of the image for analysis
		# TODO: Should we change this to accept a user input?
		input_image_path = 'contour/1.jpg'
		
		### LOAD/PREPROCESS IMAGES ###

		# Load the image for measurements
		input_image = cv2.imread(input_image_path)
		
		### MEASUREMENTS ###
		
		## Coin Measurement ##
		
		# Get the radius of the found coin in pixels
		coin_circle, _ = detect_coin(input_image)
		coin_position_x, coin_position_y, coin_radius_px = coin_circle
		
		# Calculate the pixels-per-millimetre ratio
		pixels_per_millimetre_ratio = calculate_pixels_per_millimetre_ratio(coin_radius_px, COIN_RADIUS_MM)
		
		# Get the area of the coin in pixels^2
		coin_area_px = get_circle_area_px(coin_radius_px)
		
		# Convert the coin area to millimetres^2
		coin_area_mm = get_circle_area_mm(coin_area_px, pixels_per_millimetre_ratio)
		
		# Print coin measurements
		print()
		print(f'Pixels-per-millimetre Ratio: {pixels_per_millimetre_ratio}')
		print()
		print(f'Coin Radius: {"{:.2f}px".format(coin_radius_px)}, {"{:.2f}mm".format(COIN_RADIUS_MM)}')
		print(f'Coin Area: {"{:.2f}px^2".format(coin_area_px)}, {"{:.2f}mm^2".format(coin_area_mm)}')
		
		## Wound Measurement ##
		
		# Extract the contour from the input image with a blue outline drawn around the wound
		wound_contour = extract_contour_from_outlined_image(input_image)
		
		# Wound measurements in pixels
		wound_area_px = get_contour_area_px(wound_contour)
		wound_length_x_px, wound_length_y_px = get_contour_size_px(wound_contour)
		
		# Convert the wound measurements to millimetres
		wound_area_mm = get_contour_area_mm(wound_contour, pixels_per_millimetre_ratio)
		wound_length_x_mm, wound_length_y_mm = get_contour_size_mm(wound_contour, pixels_per_millimetre_ratio)
		
		# Print wound measurements
		print()
		print(f'Wound Length X: {"{:.2f}px".format(wound_length_x_px)}, {"{:.2f}mm".format(wound_length_x_mm)}')
		print(f'Wound Length Y: {"{:.2f}px".format(wound_length_y_px)}, {"{:.2f}mm".format(wound_length_y_mm)}')
		print(f'Wound Area: {"{:.2f}px^2".format(wound_area_px)}, {"{:.2f}mm^2".format(wound_area_mm)}')
		
		### VISUALISATIONS ###
		
		## Visualise Areas ##
		
		# Make a copy of the original image
		image_areas = input_image.copy()

		# Draw coin area visualisation
		image_areas = visualise_circle_area_mm(image_areas, coin_circle, coin_area_mm)

		# Draw wound area visualisation
		image_areas = visualise_contour_area_mm(image_areas, wound_contour, wound_area_mm)

		## Visualise Lengths ##

		# Make a copy of the original image
		image_lengths = input_image.copy()

		# Draw coin length visualisation
		# TODO: Current visualisation is diamater. Function also exists to visualise radius. Confirm which is desirable.
		image_lengths = visualise_circle_diameter_mm(image_lengths, coin_circle, COIN_RADIUS_MM * 2)

		# Draw wound lengths visualisation
		image_lengths = visualise_contour_size_mm(image_lengths, wound_contour, wound_length_x_mm, wound_length_y_mm)

		## Display Visualisations ##

		# Display visualisations
		cv2.imshow('Area Measurements', image_areas)
		cv2.imshow('Length Measurements', image_lengths)

		# Wait for keypress, then close all image windows
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		print()
		print('------------------------------------------')
		print()
		print('    PRESS ENTER TO RETURN TO MAIN MENU    ')
		input()
		print('------------------------------------------')
		
		#cv2.destroyAllWindows()
		break

# Function for the "About" menu selection.
# Provides a description of the program and lists the project contributors.
def task_about():
	print('------------------------------------------')
	print('                  ABOUT                   ')
	print('------------------------------------------')
	print()
	print('     TODO: Describe the program here      ')
	print()
	print('------------------------------------------')
	print('           PROJECT CONTRIBUTORS           ')
	print('------------------------------------------')
	print()
	print('  Lucas Qin                    103527269  ')
	print()
	print('  Andrew Oates                 2473399    ')
	print()
	print('  Mona Ma                      103699029  ')
	print()
	print('  Natalie Huang                103698990  ')
	print()
	print('  Hariesh Kumar Ravichandran   104008697  ')
	print()
	print('------------------------------------------')
	print()
	print('    PRESS ENTER TO RETURN TO MAIN MENU    ')
	input()
	print('------------------------------------------')

### TASK SELECTOR ###

# Array of tasks for main menu
# [title_string, function_name]
# When item is selected its function will be called
tasks = [
			['Wound Analysis using Image Processing', task_wound_analysis_image_processing],
			['Wound Analysis using Machine Learning', None],
			['Train AI Model', None],
			['About', task_about],
			['Quit Program', quit]
		]

def get_model_model_modified_date():
	try:
		model_modified_date = os.path.getmtime(model_file_name)
		return datetime.fromtimestamp(model_modified_date).strftime('%Y-%m-%d @ %H:%M:%S')
	except FileNotFoundError:
		return 'NEVER'

def main():
	print()
	
	# Simple program loop
	# Lists names of all tasks, requests user input, calls corresponding function
	while(True):
		print('*** TITLE ***') # TODO: What is our program called?
		print()
		
		# Gives some information about the mode if one exists
		print(f'MODEL TRAINED: {get_model_model_modified_date()}')
		print()
	
		print('Select a task:')
		
		for i, task in enumerate(tasks):
			print('    {0}. {1}'.format(str(i + 1), task[0]))
		
		print()
		print('Please input a number corresponding to one of the tasks...')
		print('>', end=' ')

		selection_str = input()
		
		print()
		#print('------------------------------------------')
		#print()
		
		if selection_str.isnumeric():
			selection_int = int(selection_str) - 1
			
			if selection_int < len(tasks) and selection_int >= 0:
				if tasks[selection_int][1] is not None:
					tasks[selection_int][1]()
					print()
				else:
					print('------------------------------------------')
					print()
					print('The task "{0}" has not yet been implemented...'.format(tasks[selection_int][0]))
					print()
					print('------------------------------------------')
					print()
				
			else:
				print('------------------------------------------')
				print()
				print('ERROR! No task corresponding to input:', selection_str)
				print()
				print('------------------------------------------')
				print()
			
		else:
			print('------------------------------------------')
			print()
			print('ERROR! No task corresponding to input:', selection_str)
			print()
			print('------------------------------------------')
			print()
			
if __name__ == "__main__":
	main()