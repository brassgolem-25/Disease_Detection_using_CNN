from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Pneumonia Detection/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')




def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
                 img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Scale pixel values to [0, 1]
    img = img.astype(np.float32) / 255
    
    # # Apply random horizontal flip
    if np.random.random() < 0.5:
        img = cv2.flip(img, 1)
    img=np.expand_dims( img,axis=0 )

    preds = model.predict(img)
    # print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # pred_class = str(pred_class)               # Convert to string
        print(preds)
        print(str(preds[0][0]*100));
        if preds[0]>0.9:
            result ='This is a Pneumonia case with % '+'of '+ '  '+ str(preds[0][0]*100) 
        else:
            result='This is a Normal Case'
        return result
    return None


if __name__ == '__main__':
    app.run(port=5000, debug=True)

    # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()