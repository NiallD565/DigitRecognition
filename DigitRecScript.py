from PIL import Image
import glob
import numpy as np
import os
from os.path import basename

image_label_list = []
for filename in glob.glob('Dataset/*.png'): #assuming png
    im = Image.open(filename)
    label = (filename.split("_")[1])
    #print(label)
    image_label_list.append(label)
    im.close()

#print(image_label_list)

%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

flat_Image_List = [] # new list containing flatten image arrays
for filename in glob.glob('Dataset/*.png'):# iterate over the images contained in the directory
    im = Image.open(filename)
    img = np.reshape(im, (12288))
    flat_Image_List.append(img) # Append to the list 
    #plt.imshow(img)
    im.close()
    #print(flat_Image_List)
#print(flat_Image_List)

imageSize = 12288

from numpy import array

image_array = array(flat_Image_List)

#print(flat_Image_List)
print(image_array)

image_array = np.reshape(image_array,(len(image_array), 64, 64, 3))
image_array = image_array.astype('float32')
# Converts RGB to a range between 0-1
image_array = image_array/255

noLabels = len(np.unique(image_label_list))
print(noLabels)
label_array = np.array(image_label_list)
print(label_array.shape)

from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = ['0', '1', '2','3', '4','5', '6','7', '8', '9', 'A', 'B', 'C', 'D', 'E']

values = array(data)
print(values)

# integer encode
label_encoder = LabelEncoder()
labelData = label_encoder.fit_transform(label_array)
print(labelData)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
labelData = labelData.reshape(len(labelData), 1)
labelData = onehot_encoder.fit_transform(labelData)
print(labelData)

# invert example
inverted = label_encoder.inverse_transform([argmax(labelData[10, :])])
print(inverted)

label_array.shape
print(label_array)

input_shape = (64, 64, 3)

import random
random.Random(4).shuffle(image_array)
random.Random(4).shuffle(labelData)
#print(image_array)
#print(labelData)

from sklearn.model_selection import train_test_split
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(image_array, labelData, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# ------- MODEL -------
# Import keras.
import keras as kr
import tensorflow as tf

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Adapted from: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d?fbclid=IwAR01njT_lhc2ZZOySJnWhmq8z9iWUcKjefacuRj_bI1rJbmR0NCW1cr-ao4
# Start a neural network, building it by layers.
model = kr.models.Sequential()
# Add a hidden layer with 784 neurons.
model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(256, activation=tf.nn.relu))
model.add(Dropout(0.4))
model.add(Dense(50,activation=tf.nn.relu))

# Add a 15 neuron output layer.
model.add(Dense(15, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Number of Epoch is the amount of times the training set is put through the model
# The batch size is the amount of images the models processes at one time
model.fit(X_train,y_train, epochs=3, batch_size=300)

# ------ Testing the model ------
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Error percentage: %.2f%%" %(100 - scores[1] * 100))