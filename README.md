# RNN-ActivityRecognition

This is a RNN prototype to classify the activities drink and eat meal from the
NTU RGB+D Dataset (https://github.com/shahroudy/NTURGB-D).

The input of the model are the four orientations provided in the 3D skeleton data,
for each of the 25 joints. Accordingly, every frame is represented by a 25*4 feature vector.

Make sure Tensorflow 1.3 is installed.

UPDATE: Currently reducing the size of training and testing files referenced by model.py.
