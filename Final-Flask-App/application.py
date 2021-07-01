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
import process_wav as pv
import os


application=Flask(__name__)#create instance on flask
model=pickle.load(open('regmodel.pkl','rb'))
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':

        f1 = request.files['file-1']
        full_filename = os.path.join(application.config['UPLOAD_FOLDER'], 'file-1.m4a')
        f1.save(full_filename)

        f2 = request.files['file-2']
        full_filename = os.path.join(application.config['UPLOAD_FOLDER'], 'file-2.m4a')
        f2.save(full_filename)
        
        return render_template('index.html', upload_text='Upload Succesful')

@application.route('/verify', methods=["POST"])
def verify():
    output=True
    pv.convert()

    # deepxi_cdup = os.chdir("DeepXi-master")
    # run_deepxi = os.system("./run.sh VER='mhanet-1.1c' INFER=1 GAIN='mmse-lsa'")
    # cdintoflask = os.chdir("..")

    # print('cdup: ', deepxi_cdup)
    # print('runnn: ', run_deepxi)
    
    cdup = os.chdir("voxceleb_trainer")
    run_vox = os.system("python ./trainSpeakerNet.py --eval --model ResNetSE34L --log_input True --trainfunc angleproto --save_path exps/test --eval_frames 400 --initial_model baseline_lite_ap.model")
    cdintoflask = os.chdir("..")

    print('run_vox: ', run_vox)
    print('cdintoflask: ', cdintoflask)
    # print('run_deepxi: ', run_deepxi)

    f = open("static/result.txt", "r")
    score = f.read()
    print("score in flask: ", score)
    score = float(score)
    pred_text = ""
    if (score < -0.8):
        pred_text = "Not same speaker"
    else:
        pred_text = "Same speaker"

    return render_template('index.html', prediction_text= pred_text)

if __name__ == "__main__":
#     application.run(debug=True)
    application.run(host="0.0.0.0", port=5000)
