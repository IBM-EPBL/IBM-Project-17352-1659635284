import os
import json
from flask import Flask, redirect, url_for, render_template, request, flash
from werkzeug.utils import secure_filename
app = Flask(__name__,static_folder="static")
home_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join(home_dir, "static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
data = []

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pixellib
from pixellib.tune_bg import alter_bg  
ans=""
change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
model = load_model('asl_model_97_56.h5')

import concurrent.futures
import sys
import pyttsx3
from time import sleep


def typing(text):
    for char in text:
        sleep(0.04)
        sys.stdout.write(char)
        sys.stdout.flush()

def textToSpeech(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 250)
    engine.say(text)
    engine.runAndWait()
    del engine

def parallel(text):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_tasks = {executor.submit(textToSpeech, text), executor.submit(typing, text)}
        for future in concurrent.futures.as_completed(future_tasks):
            try:
                data = future.result()
            except Exception as e:
                print(e)

# parallel("Speak this!")
# sleep(4.0)

@app.route('/',methods = ['GET'])
def hello_world():
    print("df")
    return render_template('index.html')

@app.route('/',methods = ['POST'])
def result():
    global ans
    if request.form.get('predict') == 'Predict':
        parallel(ans)
        return render_template('index.html',msg = request.files['pic'].filename,images = json.dumps(data),msg1 = ans)
    else:
        if 'pic' not in request.files:
            print(ans)
        else:
            file = request.files['pic']
            file_name = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            data.append(file_name)
            change_bg.color_bg('static/'+file.filename, colors = (255,255,255), output_image_name="colored_bg.jpg")
            
            img = cv2.imread('colored_bg.jpg',2)
            img=cv2.resize(img,(128,128))
            ret, bw_img = cv2.threshold(img,254,255,cv2.THRESH_BINARY_INV)
            cv2.imwrite("masked.jpeg",bw_img)
            img=image.load_img(r'masked.jpeg',target_size=(128,128),color_mode='grayscale')
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            pred=np.argmax(model.predict(x))
            temp=np.expand_dims(bw_img,axis=0)
            bw_img.shape
            index=['A','B','C','D','E','F','G','H','I']
            ans += index[pred] + " "
    return render_template('index.html',msg = request.files['pic'].filename,images = json.dumps(data))

if __name__ == '__main__':
   app.run()