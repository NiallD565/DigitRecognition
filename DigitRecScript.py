from PIL import Image
import glob
import numpy as np
import os
from os.path import basename
import matplotlib
import matplotlib.pyplot as plt

image_Lbl_list = []
for filename in glob.glob('Dataset/*.png'): #assuming png
    im = Image.open(filename)
    label = (filename.split("_")[1])
    image_Lbl_list.append(label)
    im.close()

#print(image_Lbl_list)
#path = basename('Images/0')
#path

#%matplotlib inline



flat_Image_List = [] # new list containing flatten image arrays
#for img in image_Lbl_list: # iterate over the images contained in the directory
for filename in glob.glob('Dataset/*.png'):
    im = Image.open(filename)
    img = np.reshape(im, (1, 49152)) # flattened each imag (3 per each pixel)
    flat_Image_List.append(img) # Append to the list 
    #plt.imshow(img)
    im.close()
    #print(flat_Image_List)

#print(flat_Image_List)

from numpy import array

flat_Image_List = array(flat_Image_List)

#print(flat_Image_List[1])

# For encoding categorical variables and pre processing.
import sklearn.preprocessing as pre

encoder = pre.LabelBinarizer()
encoder.fit(image_Lbl_list)
outputs = encoder.transform(image_Lbl_list)

#outputs[0]

inputs = flat_Image_List.reshape(73055, 49152)

# ------- MODEL -------
# Import keras.
import tensorflow as tf
from tensorflow import keras as kr
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Start a neural network, building it by layers.
model = kr.models.Sequential()
# Add a hidden layer with 784 neurons.
model.add(kr.layers.Dense(units=784, activation='relu', input_dim=49152))
# Add a hidden layer with 455 neurons.
model.add(kr.layers.Dense(units=455, activation='relu'))
# Add a hidden layer with 170 neurons.
model.add(kr.layers.Dense(units=170, activation='softplus'))
# Add a hidden layer with 50 neurons.
model.add(kr.layers.Dense(units=50, activation='relu'))

# Add a three neuron output layer.
model.add(kr.layers.Dense(units=15, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Number of Epoch is the amount of times the training set is put through the model
# The batch size is the amount of images the models processes at one time
model.fit(inputs, outputs, epochs=3, batch_size=200)

#flat_Image_List = np.array(list(flat_Image_List[:])).reshape(flat_Image_List, 100, 49152).astype(np.uint8)
#image_Lbl_list =  np.array(list(image_Lbl_list[:])).astype(np.uint8)