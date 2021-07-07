#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:32:03 2021

@author: Ethan, Abhi, Anthony
"""


# Import libraries and external files
from flask import Flask, request, jsonify, render_template
import process_wav as pv
import os

#create instance on flask
application = Flask(__name__)

# Define the root folder path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define upload folder for m4a files to be the static folder
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Define threshold variable
THRESHOLD = -0.895

# Default homepage route
@application.route('/')
def home():
    return render_template('index.html')

# Uploadeer route
@application.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':

        # Get audio file from HTML and save to static folder as file-1.m4a
        f1 = request.files['file-1']
        full_filename = os.path.join(application.config['UPLOAD_FOLDER'], 'file-1.m4a')
        f1.save(full_filename)
        
        # Get second audio file from HTML and save to static folder as file-2.m4a
        f2 = request.files['file-2']
        full_filename = os.path.join(application.config['UPLOAD_FOLDER'], 'file-2.m4a')
        f2.save(full_filename)
        
        #Render homepage and display upload text
        return render_template('index.html', upload_text='Upload Succesful')

# Audio choices route
@application.route('/audiochoices', methods=["GET","POST"])
def audiochoices():
    num_choices = 0
    choices = []
    valid_choices = []

    # Find which audio files were checked
    if request.method == 'POST':
        a1 = request.form.get('earthamae1')
        if (a1 == 'earthamae1'):
            choices.append(a1)
        a2 = request.form.get('earthamae2')
        if (a2 == 'earthamae2'):
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

        # Determine what to display in UI
        if num_choices > 2:
            return render_template('index.html', audio_choice_text='Choose only two please ;)')
        elif num_choices < 2:
            return render_template('index.html', audio_choice_text='Choose two please ;)')
        else:
            # Open test_list file, and write appropriate file names for model to perform inference on
            f = open("voxceleb_trainer/data/test_list.txt", "w")
            f.write("1 ../static/pre_loaded_wavs/" + choices[0] + ".wav " + "../static/pre_loaded_wavs/" + choices[1] + ".wav")
            f.close()

            #Render same homepage and update audio choice text
            return render_template('index.html', audio_choice_text='Choices submitted!')

@application.route('/verifypreloaded', methods=["POST"])
def verifypreloaded():

    # Run OS commands to change directory to voxceleb directory
    cdup = os.chdir("voxceleb_trainer")

    # Run python command to perform inference on the files defined in the test_list.txt file
    run_vox = os.system("python ./trainSpeakerNet.py --eval --model ResNetSE34L --log_input True --trainfunc angleproto --save_path exps/test --eval_frames 400 --initial_model baseline_lite_ap.model")
    cdintoflask = os.chdir("..")

    #Open results file and read result
    f = open("static/result.txt", "r")
    score = f.read()
    print("score in flask: ", score)
    score = float(score)

    pred_text = ""
    dist_text = ""

    # If score is less than the threshold, display different speakers, else, display same speakers
    if (score < THRESHOLD):
        pred_text = "Different Speakers: "
        dist_text =  "Embeddings Euclidean Distance: " + str(-1 * score)[:6]
    else:
        pred_text = "Same Speakers: "
        dist_text =  "Embeddings Euclidean Distance: " + str(-1 * score)[:6]

    return render_template('index.html', prediction_text= pred_text, distance_text = dist_text)


@application.route('/verify', methods=["POST"])
def verify():
    output=True

    # Convert m4a files to wav
    pv.convert()

    # Write into test_list.txt files the audio files that the model will perform inference on
    f = open("voxceleb_trainer/data/test_list.txt", "w")
    f.write("1 ../static/file-1.wav " + "../static/file-2.wav")
    f.close()

    #Uncomment for denoising of uploaded audio files
    # deepxi_cdup = os.chdir("DeepXi-master")
    # run_deepxi = os.system("./run.sh VER='mhanet-1.1c' INFER=1 GAIN='mmse-lsa'")
    # cdintoflask = os.chdir("..")
    
    # Run OS commands to navigate to voxceleb directory and run python command to perform inference
    cdup = os.chdir("voxceleb_trainer")
    run_vox = os.system("python ./trainSpeakerNet.py --eval --model ResNetSE34L --log_input True --trainfunc angleproto --save_path exps/test --eval_frames 400 --initial_model baseline_lite_ap.model")
    cdintoflask = os.chdir("..")

    # Write result in resutl.txt file
    f = open("static/result.txt", "r")
    score = f.read()
    print("score in flask: ", score)
    score = float(score)

    pred_text = ""
    dist_text = ""
    # If score is less than threshold, display different speakers, else display same speakers
    if (score < THRESHOLD):
        pred_text = "Different Speakers: "
        dist_text =  "Embeddings Euclidean Distance: " + str(-1 * score)[:6]
    else:
        pred_text = "Same Speakers: "
        dist_text =  "Embeddings Euclidean Distance: " + str(-1 * score)[:6]

    return render_template('index.html', prediction_text= pred_text, distance_text = dist_text)

if __name__ == "__main__":
    # application.run(debug=False)
    application.run(host="0.0.0.0", debug=False)
