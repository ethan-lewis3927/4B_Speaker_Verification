#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:32:03 2021

@author: sohini
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from werkzeug import secure_filename
import process_wav as pv
import os


app=Flask(__name__)#create instance on flask
model=pickle.load(open('regmodel.pkl','rb'))
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':

        f1 = request.files['file-1']
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'file-1.m4a')
        f1.save(full_filename)

        f2 = request.files['file-2']
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'file-2.m4a')
        f2.save(full_filename)
        
        return render_template('index.html', upload_text='Upload Succesful')

@app.route('/verify', methods=["POST"])
def verify():
    output=True
    pv.convert()
    tensor1 = pv.loadWAV('static/file-1.wav', 100)
    tensor1 = pv.loadWAV('static/file-2.wav', 100)
    return render_template('index.html', prediction_text='Same Speaker: {}'.format(output))

@app.route('/predict', methods=["POST"])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)
    return render_template('index.html', prediction_text='Profit should be ${}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
