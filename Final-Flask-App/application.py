#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:32:03 2021

@author: sohini, ethan, ahbi, anthony
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import process_wav as pv
import os


application = Flask(__name__)#create instance on flask
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

@application.route('/audiochoices', methods=["GET","POST"])
def audiochoices():
    num_choices = 0
    choices = []
    valid_choices = []

    # Find which audio files were checked
    if request.method == 'POST':
        a1 = request.form.get('jonnydep1')
        if (a1 == 'jonnydep1'):
            choices.append(a1)
        a2 = request.form.get('jonnydep2')
        if (a2 == 'jonnydep2'):
            choices.append(a2)
        a3 = request.form.get('denzelwashinton1')
        if (a3 == 'denzelwashinton1'):
            choices.append(a3)
        a4 = request.form.get('denzelwashinton2')
        if (a4 == 'denzelwashinton2'):
            choices.append(a4)
        for choice in choices:
            if choice != None:
                num_choices += 1
                valid_choices.append(choice)

        print(choices)
        # Determine what to display in UI
        if num_choices > 2:
            return render_template('index.html', audio_choice_text='Choose only two please ;)')
        elif num_choices < 2:
            return render_template('index.html', audio_choice_text='Choose two please ;)')
        else:
            f = open("voxceleb_trainer/data/test_list.txt", "w")
            f.write("1 ../static/pre_loaded_wavs/" + choices[0] + ".wav " + "../static/pre_loaded_wavs/" + choices[1] + ".wav")
            f.close()
            return render_template('index.html', audio_choice_text='Choices submitted!')

@application.route('/verifypreloaded', methods=["POST"])
def verifypreloaded():
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
        pred_text = "Not same speaker, score: " + str(score)[:6]
    else:
        pred_text = "Same speaker, score: "  + str(score)[:6]

    return render_template('index.html', prediction_text= pred_text)

@application.route('/verify', methods=["POST"])
def verify():
    output=True
    pv.convert()

    f = open("voxceleb_trainer/data/test_list.txt", "w")
    f.write("1 ../static/file-1.wav " + "../static/file-2.wav")
    f.close()

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
        pred_text = "Not same speaker, score: " + str(score)[6]
    else:
        pred_text = "Same speaker, score: " + str(score)[6]

    return render_template('index.html', prediction_text= pred_text)

if __name__ == "__main__":
    # application.run(debug=False)
    application.run(host="0.0.0.0", debug=False)
