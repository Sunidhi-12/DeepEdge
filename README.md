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

The rationale behind the dataset choices is as follows:

Unique Coordinate Representation: Each sample in the dataset represents a unique (x, y) coordinate pair in a 50x50 pixel image. This choice ensures that the dataset covers all possible coordinates within the image space, allowing the model to learn the relationship between the coordinates and the corresponding pixel values.

Dataset Size: The dataset size is determined by the total number of unique coordinates within the 50x50 pixel image space. Since each sample represents a unique coordinate, the dataset size is equal to the total number of unique coordinates. In this case, there are 50 * 50 = 2500 unique coordinates, so the dataset consists of 2500 samples.

Dataset Diversity: While each sample is unique in terms of its (x, y) coordinate pair, the dataset does not incorporate data augmentation techniques. This is because data augmentation is not applicable or beneficial for this specific problem, where each sample already covers all possible scenarios. Therefore, the focus is on ensuring diversity through the coverage of all unique coordinates rather than introducing variability through augmentation.
