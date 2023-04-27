import os
import cv2

from datetime import datetime

#from v1_coin import display_coin_detection, detect_coin, calculate_wound_area, estimate_actual_area

training_image_paths = ['fake_wound/']
training_mask_paths = ['fake_jj/']

testing_image_paths = []

model_file_name = 'wound_segmentation_model.h5'

COIN_RADIUS = 14.25 # Radius of an Australian $2 coin in millimeters

## MODEL TRAINING

def dice_coefficient(y_true, y_pred):
	y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
	y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
	intersection = tf.reduce_sum(y_true_f * y_pred_f)
	return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

def convert_image_for_display(image):
		return np.uint8(image[:, :, :3])

def model_train():
	X, y = load_images_and_masks(images_json_path, images_png_path, masks_json_path, masks_png_path)

	# Data augmentation
	data_gen_args = dict(rotation_range=20,             # Increase rotation range
							width_shift_range=0.1,         # Increase width shift range
							height_shift_range=0.1,        # Increase height shift range
							shear_range=0.1,               # Increase shear range
							zoom_range=0.1,                # Increase zoom range
							brightness_range=(0.9, 1.1),
							horizontal_flip=True,
							vertical_flip=True,            # Add vertical flipping
							fill_mode='nearest')
	
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	
	# Compile the model
	model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[dice_coefficient])
	
	# Train the model
	batch_size = 8  

	train_generator = zip(image_datagen.flow(X_train, batch_size=batch_size, seed=42),
							mask_datagen.flow(y_train, batch_size=batch_size, seed=42))

	val_generator = zip(image_datagen.flow(X_val, batch_size=batch_size, seed=42),
							mask_datagen.flow(y_val, batch_size=batch_size, seed=42))

	early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

	model.fit(train_generator, steps_per_epoch=len(X_train) // batch_size, validation_data=val_generator,
				validation_steps=len(X_val) // batch_size, epochs=1)
	
	# Save the model
	model.save(model_file_name)

## MODEL PREDICTION

def model_predict():
	# Check to see if the model file exists
	if os.path.isfile(model_file_name):
		# Load the model
		model = load_model(model_file_name)
		
		additional_input = True  
		evaluation_images = load_evaluation_images(evaluation_path, additional_input)
		
		# Predict and display results
		predicted_masks = model.predict(evaluation_images)
		
		# Set a threshold value
		threshold = 0.6
		num_clusters = 8
		
		# Apply threshold to the predicted masks
		binary_masks = (predicted_masks > threshold).astype(np.uint8) * 255
	
		return binary_masks
	else:
		print(f'ERROR: File "{model_file_name}" not found...')
		return None

## WOUND MESAUREMENT

def wound_measurement():
	pass

## COLOUR ANALYSIS

def colour_analysis():
	pass

## TASK SELECTOR

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

def get_model_creation_date():
	try:
		creation_date = os.path.getmtime(model_file_name)
		return datetime.fromtimestamp(creation_date).strftime('%Y-%m-%d @ %H:%M:%S')
	except FileNotFoundError:
		return 'NEVER'

def measure_wound():
	image = cv2.imread(images_path)
	detect_coin(image, min_radius=30, max_radius=100)

# Array of tasks for main menu
# [string_title, function_name]
# When item is selected its function will be called
tasks = [
			['About', task_about],
			['Train Model', None],
			['Find Wound Area', None],
			['Measure Wound', None],
			['Wound Colour Analysis', None],
			['Quit Program', quit]
		]

def main():
	print()
	
	# Simple program loop
	# Lists names all tasks, requests user input, calls corresponding function
	while(True):
		print('*** TITLE ***')
		print()
		
		# Gives some information about the mode if one exists
		print(f'MODEL TRAINED: {get_model_creation_date()}')
		print()
	
		print('Select a task:')
		
		for i, task in enumerate(tasks):
			print('    ' + str(i) + '. ' + str(task[0]))
		
		print()
		print('Please input a number corresponding to one of the tasks...')
		print('>', end=' ')

		selection = input()
		
		print()
		
		if selection.isnumeric():
			selection = int(selection)
			
			if selection < len(tasks) and selection >= 0:
				if tasks[selection][1] is not None:
					tasks[selection][1]()
					print()
				else:
					print('The task "' + str(tasks[selection][0]) + '" has not yet been implemented...')
					print()
					
			else:
				print('ERROR: No task corresponding to input: ' + str(selection))
				print()
				
		else:
			print('ERROR: No task corresponding to input: ' + str(selection))
			print()
			
if __name__ == "__main__":
	main()