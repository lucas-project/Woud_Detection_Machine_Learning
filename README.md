Version_1.py is a methods of implementing border detection and colour detection. It used Tensorflow as framework, over 300 artifact and real wound images as training materials to create a model that is able to identify wound's border and its change.

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

All training materials need to add mask manually using Labelbox
![masks](https://github.com/lucas-project/Woud_Detection_Machine_Learning/assets/87470079/f0b2d4b3-1b98-4cb1-ba30-c7c530130982)

The mask that is ready for use
![mask-2](https://github.com/lucas-project/Woud_Detection_Machine_Learning/assets/87470079/8b223837-7543-4c19-8be2-b434f4502364)

Wound image, model's identification result, and the result after fine tune
![wound-result-fine tune result](https://github.com/lucas-project/Woud_Detection_Machine_Learning/assets/87470079/be4969db-913f-44e2-b01b-35cf88d16f3b)

Evaluation dataset samples
![evaluation dataset](https://github.com/lucas-project/Woud_Detection_Machine_Learning/assets/87470079/2fe266f2-f051-4230-94fa-dfd76a205915)

different type of training materials used
![different trained artifact wound images](https://github.com/lucas-project/Woud_Detection_Machine_Learning/assets/87470079/1a5d76b0-7875-429c-abf7-0d397c4f36dc)

Training and validation loss curve 
![training and validation loss curve](https://github.com/lucas-project/Woud_Detection_Machine_Learning/assets/87470079/91e2fb87-9d14-4d30-a513-db1e6f75065b)

In case of complicated wound, doctor have the option to circle out wound's border manually, blue line is the border marked manually by a doctor, and the black and white images show the identification result by our algorithms.
![manual circle out complicated wound](https://github.com/lucas-project/Woud_Detection_Machine_Learning/assets/87470079/5f475311-0171-497d-8b98-c75fb484e7f9)
