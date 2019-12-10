# Behavioral Cloning

## Overview

The project aims to train a deep learning neural network that learns how to drive from the user interaction with the simulator and then it clones the behaviour of the user. The data set that was used for the training part was supplied by the udacity team and it consists in a series of images and control parameters.

The main parts of the code:

1. *Reading the data from the dataset*
2. *Split the training data from the validation data*
3. *Crop the images only to a region of interest*
4. *Normalize the images*
5. *Implement NVIDIA neural network model*
6. *Train and see the accuracy*

[//]: # (Image References)

[image1]: error_loss.png "Loss Visualization"

## The code

The primary code is written in the "model.py" file. There you'll can see the implementation of the network architecture and the model training.
After the training, the weights of the model are saved into the "model.h5" file and it can be reused later.

## Model Architecture and Training Strategy

The architecture used for the neural network was the one proposed by [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
It has an input size of 320 x 80 x 3 and a regression output of 1.
I have implemented 2 preprocessing methods:
- The first one is the cropping of the original images remaining with the region of interest in our images.
- The second one is data normalizing implemented using keras Lambda layer.

For reducing the overfitting I used a dropout of 0.2 after each of the fully connected layers.
The loss function implemented here is Mean Square Error and the optimizer is Adam Optimizer (better than Gradient Descend).

I ploted the training and validation loss for each epoch.
![alt text][image1] 
