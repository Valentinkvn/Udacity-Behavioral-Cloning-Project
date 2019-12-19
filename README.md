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

[image1]: ./readme_images/error_loss.png "Loss Visualization"
[image2]: ./readme_images/original_image.png "Original image"
[image3]: ./readme_images/cropped_image.png "Cropped image"
[image4]: ./readme_images/three_images.png "All three images"
[image5]: ./readme_images/model_architecture.png "Model Architecture"

## The code

The primary code is written in the "model.py" file. There you'll be able to see the implementation of the network architecture and the model training.
After the training, the weights of the model are saved into the "model.h5" file and it can be reused later.
Also, in the "drive.py" file you are able to see the python routine that manages the controlling of the car. There is also a simple PI control implementation which is used to smooth the speed changings to a user input set value.

## Model Architecture and Training Strategy

The images in the dataset were taken using three cameras, each of these were mounted on left, center and right part of the car.

![alt text][image4] 

The architecture used for the neural network was the one proposed by [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

<p align="center">
  <img src="/readme_images/model_architecture.png">
</p>

The original images were cropped to match the input size of our network.

<p align="center">
  <img src="/readme_images/original_image.png">
</p>

<p align="center">
  <img src="/readme_images/cropped_image.png">
</p>

It has an input size of 320 x 80 x 3 and a regression output of 1 (the steering angle).
I have implemented 2 preprocessing methods:
- The first one is the cropping of the original images remaining with the region of interest in our images (we don't need the top and the bottom of the images).
- The second one is data normalizing implemented using keras Lambda layer (for a better convergence of the network).

For reducing the overfitting I used a dropout of 0.2 after each of the fully connected layers.
The loss function implemented here is Mean Square Error and the optimizer is Adam Optimizer (better than Gradient Descend).

#### Visualization of error loss

I ploted the training and validation loss for each epoch.

<p align="center">
  <img src="/readme_images/error_loss.png">
</p>

#### Demo

Here is a reprezentation of the working behavioral cloning.

<p align="center">
  <img src="/readme_images/gif.gif">
</p>
