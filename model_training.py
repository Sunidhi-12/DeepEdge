#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import pickle
# Loadinf the dataset
def load_dataset(load_path):
    data = np.load(load_path)
    images = data['images']
    labels = data['labels']
    return images, labels

def main():
    dataset_path = "dataset.npz"
    X, y = load_dataset(dataset_path)
    # Splitting the dataset into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2, random_state=101)
    # Model architecture
    model = tf.keras.Sequential([
        Input(shape=(50,50,1)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2)
    ])
    model.compile(metrics='accuracy',optimizer='adam',loss='mse')
    model.fit(X_train, y_train, validation_data = (X_val,  y_val), epochs=10, batch_size=32)
    # Model saving
    model.save('model.h5')
if __name__ == "__main__":
    main()

