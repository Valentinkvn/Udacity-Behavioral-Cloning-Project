# import libraries
import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import Dropout
import matplotlib.pyplot as plt

# data set that is used for the training
data_set = 'data'

# variable used for storing informations about images
samples = []
with open(data_set + '/driving_log.csv') as csvfile:
    print("Dataset " + str(data_set) + " loaded.")
    reader = csv.reader(csvfile)
    line_number = 0
    for line in reader:
        if line_number > 0: 

            # file path to all 3 images
            center_img = line[0]
            left_img = line[1]
            right_img = line[2]
            
            # parameter for the corection of steering regarding different perspective
            correction = 0.2
            
            steering_center = float(line[3])
            steering_left = steering_center + correction
            steering_right = steering_center - correction    
            
            # a tuple of data
            center_sample = [center_img, steering_center]
            left_sample = [left_img, steering_left]
            right_sample = [right_img, steering_right]       
            
            samples.extend([center_sample, left_sample, right_sample])
            
        line_number += 1

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("The number of training samples is: " + str(len(train_samples)))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                img_name = data_set + '/IMG/' + batch_sample[0].split('/')[-1]
                
                # open image at the file path
                image = cv2.imread(img_name)
                steering = float(batch_sample[1])
                
                # append images and steering angle
                images.append(image)
                angles.append(steering)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


            
# Set our batch size
batch_size = 32
# Trimmed image format
ch, row, col = 3, 80, 320  

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Create the model
model = Sequential()
# Crop image on top and bottom 
model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape=(160, 320, 3)))
# Preprocessincoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
# 5 convolution layers
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu", border_mode='valid'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu", border_mode='valid'))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu", border_mode='valid'))
model.add(Convolution2D(64, 3, 3, activation = "relu", border_mode='valid'))
model.add(Convolution2D(64, 3, 3, activation = "relu", border_mode='valid'))
model.add(Flatten())
# 3 fully connected layers with relu activation and a dropout of 0.2
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


# Used mean squred error and adam optimzer
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch = np.ceil(len(train_samples)/batch_size), validation_data = validation_generator, validation_steps = np.ceil(len(validation_samples)/batch_size), epochs = 5, verbose = 1)

### print the keys contained in the history object
# print(history_object.history.keys())

#### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')