from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle
import logging

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/finalized_model.h5'

#Loading trained model using pickle
model = pickle.load(open(MODEL_PATH, 'rb'))

#Query model to make a prediction
model._make_predict_function()       
#print('Model loaded. Start serving...')

#Function to do model prediction
def model_predict(img_path, model):
    #load image with taget  size of 64 x 64 pixels. 
    #This is what out model is trained on.
    img = image.load_img(img_path, target_size=(64, 64))
    # Preprocessing the image, convert image to array
    x = image.img_to_array(img)
    # Reshape the image to expected size that model is trained on
    x = np.reshape(x,(1, 64, 64, 3))
    x = x.astype('float32')
    x = x/255

    # Get model to make prediction and return results
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

# Displays the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Requests the prediction request to the model
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Handle the labels of the images
        label_encoder = LabelEncoder()
        data = ['0', '1', '2','3', '4','5', '6','7', '8', '9', 'A', 'B', 'C', 'D', 'E']
        tempLabelData = label_encoder.fit_transform(data)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for user
        TempResult = preds.argmax(axis=-1)
        # Convert result to inverse to get readable result for user
        result = str(label_encoder.inverse_transform(TempResult))
        return result[2]
       
    return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()