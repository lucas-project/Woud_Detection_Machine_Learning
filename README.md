Version_1.py and Version_2.py are 2 different and independent methods of implementing border detection and colour detection.

#Version_1.py
Version_1 used TensorFlow with Kera, U-net architecture with VGG16 encoder

To run the code:
Install required libraries first:
'''
pip install numpy opencv-python tensorflow requests pillow scikit-learn scikit-image matplotlib
'''
Then:
'''
python version_1.py
'''

'version_1.py': Main program
'v1_border': functions related to border identification
'v1_colour': functions related to colour handling
'v1_evaluation': functions realted to evaluate training performance

'json_images' folder: images with masking in JSON format
'json_masking' folder: JSON format masking
'jj_images' folder: images with masking in .jpg format
'jj_masking' folder: .jpg format masking
'evaluation' folder: images for evaluating model's performance

'fake_jj' folder: JSON format masking 
'fake_wound' folder: fake images with .jpg format maksing
'fake_evaluation' folder: fake evaluation images

'wound_segmentation_model.h5': the model file

pip install opencv-contrib-python