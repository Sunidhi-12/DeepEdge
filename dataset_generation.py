#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
# Dataset generation using random integer
def dataset_generation():
    image = np.zeros((50, 50))
    x = np.random.randint(0, 50)
    y = np.random.randint(0, 50)
    image[x, y] = 255
    image = image / 255.0
    return image, (x, y)

def main():
    total_samples = 2500
    images = []
    labels = []
    # Making sure that dataset contains all unique samples
    unique_samples = set()

    while len(images) < total_samples:
        image, label = dataset_generation()
        if label not in unique_samples:
            images.append(image)
            labels.append(label)
            unique_samples.add(label)

    images = np.array(images)
    labels = np.array(labels)
    # Saving the dataset
    np.savez("dataset.npz", images=images, labels=labels)

if __name__ == "__main__":
    main()

