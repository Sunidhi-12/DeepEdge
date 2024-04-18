# DeepEdge
Pixel Coordinate Prediction 

This repository contains scripts for training and validating a deep learning model to predict the coordinates (x, y) of a pixel with a value of 255 in a given 50x50 pixel grayscale image.

Dependencies
To run the scripts, you need the following dependencies:

Python 3.x
TensorFlow
NumPy
Matplotlib (for visualization)

Step 1:
Run dataset_generation.py
This is will create the dataset and download the dataset.

Step 2: 
Run model_training.py
This will train the model on the dataset created and download the model.h5.

Step 3:
Run the prediction.py 
This will predict for test data and download a jpg of scatter plot of prediction vs actual values.
