### Integrate HTML With Flask
### HTTP verb GET And POST
from flask import Flask,redirect,url_for,render_template,request
import math
import cv2
import numpy as np
from time import time
import random
import mediapipe as mp
import matplotlib.pyplot as plt

app=Flask(__name__)

# @app.route('/')
# def welcome():
#     return render_template('index_f.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/bmi_final.html')
def bmi():
    return render_template('bmi_final.html')

@app.route('/about.html')
def About():
    return render_template('about.html')

@app.route('/Calorie-Counter.html')
def CalCount():
    return render_template('Calorie-Counter.html')

@app.route('/games.html')
def Game():
    return render_template('games.html')

@app.route('/blog.html')
def blog():
    return render_template('blog.html')

@app.route('/video.html')
def vid():
    return render_template('video.html')

@app.route('/blog-post-1.html')
def blog1():
    return render_template('blog-post-1.html')

@app.route('/video-post-1.html')
def vid1():
    return render_template('video-post-1.html')

@app.route('/blog-post-2.html')
def blog2():
    return render_template('blog-post-2.html')

@app.route('/video-post-2.html')
def vid2():
    return render_template('video-post-3.html')

@app.route('/blog-post-3.html')
def blog3():
    return render_template('blog-post-3.html')

@app.route('/video-post-3.html')
def vid3():
    return render_template('video-post-3.html')

@app.route('/blog-post-4.html')
def blog4():
    return render_template('blog-post-4.html')

@app.route('/video-post-4.html')
def vid4():
    return render_template('video-post-4.html')

@app.route('/blog-post-5.html')
def blog5():
    return render_template('blog-post-5.html')

@app.route('/video-post-5.html')
def vid5():
    return render_template('video-post-5.html')

@app.route('/blog-post-6.html')
def blog6():
    return render_template('blog-post-6.html')

@app.route('/video-post-6.html')
def vid6():
    return render_template('video-post-6.html')

import requests
import json

@app.route('/submit', methods=['POST'])
def submit():
    output = request.form.to_dict()
    food_item = output['meal1']

    end_pt_url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    query = {
        "query": food_item,
    }
    api_id = "API ID"
    api_key = "API KEY"

    headers = {
        "x-app-id": api_id,
        "x-app-key": api_key,
        "Content-Type": "application/json"
    }

    r = requests.post(end_pt_url, headers=headers, json=query)
    data = json.loads(r.text)

    for food in data['foods']:
        name = food['food_name']
        cal = food['nf_calories']
        serving_qty = food["serving_qty"]
        serving_unit = food["serving_unit"]
        return render_template('result.html', cal=cal)

@app.route('/bmi', methods=['POST'])
def bmi_res():
    output = request.form.to_dict()
    height = int(output['height'])
    weight = int(output['weight'])
    height_ = int(height) / 100
    bmi = (weight / (height ** 2)) * 10000
    if bmi <= 18.5:
        bmi_ = 'Underweight'
        bmi__ = 'Underweight'

    elif 18.5 < bmi <= 24.9:
        bmi_ = 'Normal'
        bmi__ = 'Normal'

    elif 25 <= bmi <= 29.9:
        bmi_ = 'Overweight'
        bmi__ = 'Overweight'

    elif 30 <= bmi <= 39.9:
        bmi_ = 'Obese'
        bmi__ = 'Obese'
    elif bmi >= 40:
        bmi_ = 'Morbidly'
        bmi__ = 'Morbidly Obese'
    else:
        bmi_ = 'Incorrect'
        bmi__ = 'Incorrect input'
    return render_template('bmi.html', bmi=round(bmi,2), bmi_ = bmi_, bmi__ = bmi__)
    
@app.route('/result', methods=['POST'])
def calculate_calories_burnt():
    exercise = request.form.get('exercise')
    gender = request.form.get('gender')
    weight = float(request.form.get('weight'))
    height = float(request.form.get('height'))
    age = int(request.form.get('age'))

    # Make API request to calculate calories burnt
    end_pt_url = "https://trackapi.nutritionix.com/v2/natural/exercise"

    query = {
        "query": exercise,
        "gender": gender,
        "weight_kg": weight,
        "height_cm": height,
        "age": age
    }


    headers = {
        "x-app-id": api_id,
        "x-app-key": api_key,
        "Content-Type": "application/json"
    }

    response = requests.post(end_pt_url, headers=headers, json=query)
    data = json.loads(response.text)

    exercise_details = []
    for exercise in data['exercises']:
        name = exercise['name']
        calories_burnt = exercise['nf_calories']
        duration = exercise["duration_min"]
        exercise_details.append((name, calories_burnt, duration))

    return render_template('submit.html', calories_burnt=calories_burnt)

import nltk
nltk.download('popular')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random

intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/rapidmoves', methods=['GET'])
def open_rapidmoves():
    return render_template('InsaneAIgame.html')

@app.route('/posesurfers', methods=['GET'])
def posesurfers():
    return render_template('posesurfers.html')

@app.route('/poseperfect', methods=['GET'])
def rapidmoves():
    return render_template('poseperfect.html')
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
from flask import Flask,render_template,Response

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

from flask import Flask, render_template, Response
import time
import cv2
import imutils
from imutils.video import VideoStream
import pyautogui

# generate frames and yield to Response
def gen_frames():
    # Load the haar cascade face detector
    detector = cv2.CascadeClassifier("haarcascade_upperbody.xml")

    # Set the tracker to None
    tracker = None

    # Start the video stream through the webcam
    vs = VideoStream(src=0).start()

    # Scale factor to resize the frame for faster processing
    scale = 2

    # Height and Width from the webcam
    H = 480 // scale
    W = 640 // scale

    # Define the boundaries
    up = 160 // scale
    down = 320 // scale
    left = 200 // scale
    right = 440 // scale

    # By default each key press is followed by a 0.1 second pause
    pyautogui.PAUSE = 0.0

    # wait sometime until next movement is registered
    wait_time = 0.01
    start = end = 0

    # total number of frames processed thus far and skip frames
    totalFrames = 0
    skip_frames = 50

    # loop indefinitely
    while True:
        # grab the video frame, laterally flip it and resize it
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=W)

        # initialize the action
        action = None

        # Run the face detector to find or update face position
        if tracker is None or totalFrames % skip_frames == 0:
            # convert the frame to grayscale (haar cascades work with grayscale)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect all faces
            faces = detector.detectMultiScale(gray, scaleFactor=1.05,
                                              minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

            # Check to see if a face was found
            if len(faces) > 0:
                # Pick the most prominent face
                initBB = faces[0]

                # start the tracker
                tracker = cv2.legacy_TrackerKCF.create()
                tracker.init(frame, tuple(initBB))
            else:
                tracker = None

        # otherwise the tracker is tracking the face, update the position
        else:
            # grab the new bounding box coordinates of the face
            (success, box) = tracker.update(frame)

            # if tracking was successful, draw the center point
            if success:
                (x, y, w, h) = [int(v) for v in box]

                # calculate the center of the face
                centerX = int(x + (w / 2.0))
                centerY = int(y + (h / 2.0))

                # draw a bounding box and the center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), -1)

                # determine the action
                if centerY < up:
                    action = "up"
                elif centerY > down:
                    action = "down"
                elif centerX < left:
                    action = "left"
                elif centerX > right:
                    action = "right"

            else:
                tracker = None

        end = time.time()
        # press the key
        if action is not None and end - start > wait_time:
            pyautogui.press(action)
            start = time.time()

        # draw the lines
        cv2.line(frame, (0, up), (W, up), (255, 255, 255), 2)  # UP
        cv2.line(frame, (0, down), (W, down), (255, 255, 255), 2)  # DOWN
        cv2.line(frame, (left, up), (left, down), (255, 255, 255), 2)  # LEFT
        cv2.line(frame, (right, up), (right, down), (255, 255, 255), 2)  # RIGHT

        # increment the totalFrames and draw the action on the frame
        totalFrames += 1
        text = "{}: {}".format("Action", action)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Generate a stream of frame bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/templerush')
def temple_rush():
    return render_template('templerush.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=='__main__':
    app.run(debug=True)