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

logging.debug('d1')

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/finalized_model.h5'

model = pickle.load(open(MODEL_PATH, 'rb'))
logging.debug('d2')
# Load your trained model
#model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')
logging.debug('d3')
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(128, 128))
    img = image.load_img(img_path, target_size=(64, 64))
    #, target_size=(64, 64) 49152, 12288
    #img = np.reshape(img, (49152))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.reshape(x,(1, 64, 64, 3))
    x = x.astype('float32')
    x = x/255

    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

logging.debug('d4')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

logging.debug('d5')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

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

        # Process your result for human
        #pred_class = preds.argmax(axis=-1) # Simple argmax
        #result = str(preds) # Convert to string
        TempResult = preds.argmax(axis=-1)
        result = str(label_encoder.inverse_transform(TempResult))
        return result[2]
       
    return None

logging.debug('d6')

if __name__ == '__main__':
    app.run(debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()