#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

def dataset_generation():
    image = np.zeros((50, 50))
    x = np.random.randint(0, 50)
    y = np.random.randint(0, 50)
    image[x, y] = 255
    image = image / 255.0
    return image, (x, y)

def main():
    model = tf.keras.models.load_model('model.h5')
    test_data_size = 10 
    test_image=[]
    test_label=[]
    # Generating random test data 
    for _ in range(test_data_size):
        image_t, label_t = dataset_generation()
        test_image.append(image_t)
        test_label.append(label_t)
    test_image = np.array(test_image)
    test_label = np.array(test_label)
    # Predicting 
    preds=model.predict(test_image)
    x_pred, y_pred = preds[:,0],preds[:,1]
    x_actual, y_actual = test_label[:,0],test_label[:,1]
    # Scatterplot for the prediction vs actual value
    plt.figure(figsize=(8,6))
    plt.scatter(x_pred, y_pred, color='blue', label='Predicted Coordinates', alpha=0.5)
    plt.scatter(x_actual, y_actual, color='red', label='Actual Coordinates', alpha=0.5)
    # Scing the scatter plot as jpg
    plt.savefig('scatter_plot.jpg')
if __name__ == "__main__":
    main()

