# 4B_Speaker_Verification
Speaker verification repository for 4th Brain Capstone Project. 

### Project Structure
This project has four major parts :
1. voxceleb_trainer - This directory was cloned from the voxceleb repository: https://github.com/clovaai/voxceleb_trainer. We used the code in here to train our models and perform inference on uploaded files in our deployed application. 
2. application.py - This file contains Flask APIs that receives input from the user through the GUI or API calls then computes the verifies two audio clips as the same speaker or not and returns it. The user is either able to upload their own two m4a files and verify if the files come from the same speaker or select two of our four pre-loaded audio files. 
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to upload their own audio files or choose two pre-loaded files to verify if the audio files come from the same speaker
5. DeepXi-master - This directory was cloned from the DeepXi repository: https://github.com/anicolson/DeepXi. We used the code found here to denoise our training data and denoise audio files when performing inference. 

### Running the project
1. Refer to the voxceleb github to train or download a pretrained model to perform inference. The github can be found here: https://github.com/clovaai/voxceleb_trainer

2. Run app.py using below command to start Flask API
```
python application.py
```
By default, flask will run on http://0.0.0.0:5000/ (localhost)

3. Navigate to URL http://0.0.0.0:5000/ (localhost)
