import os
import cv2
import json
import matplotlib.pyplot as plt
from datetime import datetime

#def save_wound_data(output_path, name, image, contours, size_x_mm, size_y_mm, area_mm):
def save_wound_data(output_path, name, image, mask, wound_results):
	# Check if any wound results exist, otherwise there is nothing to save
	if wound_results:
		# Create folders
		output_subpath = os.path.join(output_path, name)
		os.makedirs(output_subpath, exist_ok=True)
		
		# Save the current date/time
		filename_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # Filename-friendly format
		
		# Save the image
		image_filename = f'wound_image_{name}_{filename_timestamp}.jpg'
		image_path = os.path.join(output_subpath, image_filename)
		cv2.imwrite(image_path, image)
		
		# Save the mask
		mask_filename = f'wound_mask_{name}_{filename_timestamp}.jpg'
		mask_path = os.path.join(output_subpath, mask_filename)
		cv2.imwrite(mask_path, mask)
		
		# Format the wound data as JSON 
		data = {}
		data['name'] = name
		data['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # More readable format
		
		data['image'] = image_path
		data['mask'] = mask_path
		data['wounds'] = []
		
		# Loop through each contour and create sub nodes for each
		#for i, contour in enumerate(contours):
		#	wound = {}
		#	wound['id'] = i
		#	wound['contour'] = contour
		#	wound['size_x'] = size_x_mm
		#	wound['size_y'] = size_y_mm
		#	wound['area'] = area_mm
		#	
		#	data['wounds'].append(wound)
		for i, result in enumerate(wound_results):
			wound = {}
			wound['id'] = i
			wound['size_x'] = result[0]
			wound['size_y'] = result[1]
			wound['area'] = result[2]
			
			data['wounds'].append(wound)
		
		# Save the JSON file
		json_filename = f'wound_result_{name}_{filename_timestamp}.json'
		
		with open(os.path.join(output_subpath, json_filename), 'w') as file:
			json.dump(data, file)
		
		print()
		print(f'Results Saved: "...\{output_subpath}"')
	else:
		print('No countours found in', contours)

def load_wound_data(path):
	# Check to see if path given has any results
	if os.path.exists(path) and os.listdir(path):
		print()
		print('Loading results...')
		# TODO: MOVE LOADING IN TO FUNCTION!
		# Load the saved data
		results = []
		
		# Loop through each file in the folder
		for file_name in os.listdir(path):
			file_path = os.path.join(path, file_name)
			
			# Check if the file is a JSON file
			if file_name.endswith('.json'):
				# Add it to the list
				with open(file_path) as file:
					results.append(json.load(file))
		
		# If any results were loaded, return the list
		if results:
			return results
		
		# Otherwise print an error and return None
		print()
		print(f'No results exist at "...\{path}"')
		
		return None
	
	else:
		print(f'No results exist at "...\{path}"')

def plot_wound_data(results):
	# Store wound data in a dictionary {id: data}
	wound_data = {}
	
	# Loop through each result
	for result in results:
		wounds = result['wounds']
		
		# Loop through each wound in the result
		for wound in wounds:
			id = wound['id']
			timestamp = datetime.strptime(result['date'], '%Y-%m-%d %H:%M:%S')
			size_x = wound['size_x']
			size_y = wound['size_y']
			area = wound['area']
			
			# If wound ID already exists in the dictionary, append the data
			if id in wound_data:
				wound_data[id]['timestamp'].append(timestamp)
				wound_data[id]['size_x'].append(size_x)
				wound_data[id]['size_y'].append(size_y)
				wound_data[id]['area'].append(area)
				
			# If the wound ID doesn't exists in the dictionary, add it
			else:
				wound_data[id] = {'timestamp': [timestamp], 'size_x': [size_x], 'size_y': [size_y], 'area': [area]}
	
	# Create the plot
	fig, ax_size = plt.subplots()
	ax_size.set_xlabel('Date')
	ax_size.set_ylabel('Wound Sizes')
	ax_size.set_title('Wound Sizes Over Time')
	
	# Create a second Y axis for the areas
	ax_area = ax_size.twinx()
	ax_area.set_ylabel('Wound Areas')
	
	# Loop through each wound in the dictionary and plot the data
	for id, data in wound_data.items():
		ax_size.plot(data['timestamp'], data['size_x'], label=f'Size X ({id})', color='green')
		ax_size.plot(data['timestamp'], data['size_y'], label=f'Size Y ({id})', color='orange')
		ax_area.plot(data['timestamp'], data['area'], label=f'Area ({id})', color='blue')
	
	# Add a legend
	ax_size.legend()
	ax_area.legend(loc='lower right')

	# Show the plot
	plt.show()