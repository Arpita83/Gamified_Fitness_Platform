### Integrate HTML With Flask
### HTTP verb GET And POST
from flask import Flask,redirect,url_for,render_template,request
import math
import cv2
import numpy as np
import time
import random
import mediapipe as mp
import matplotlib.pyplot as plt
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

@app.route('/random_rapid_moves.html')
def rand_rapid_moves():
    return render_template('random_rapidmoves.html')

# Calorie Counter (meal)

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

# bmi counter

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
    
# Calorie counter (exercise)

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
        "x-app-id": "66e88fcd",
        "x-app-key": "915bd34e3996d68e870d3be75c07b467",
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

# Fitness CHatbot

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
    return render_template('rapidmoves.html')

@app.route('/posesurfers', methods=['GET'])
def posesurfers():
    return render_template('posesurfers.html')

@app.route('/poseperfect', methods=['GET'])
def rapidmoves():
    return render_template('poseperfect.html')

@app.route('/templerush', methods=['GET'])
def templerush():
    return render_template('templerush.html')

@app.route('/drivefit', methods=['GET'])
def drivefit():
    return render_template('drivefit.html')

# camera = cv2.VideoCapture(0)
# def generate_frames():
#     while True:
            
#         ## read the camera frame
#         success,frame=camera.read()
#         if not success:
#             break
#         else:
#             ret,buffer=cv2.imencode('.jpg',frame)
#             frame=buffer.tobytes()

#         yield(b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# from flask import Flask,render_template,Response

# @app.route('/video')
# def video():
#     return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

# Temple Rush

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

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rapid Moves

import cv2
import numpy as np
from flask import Flask, render_template, Response
import mediapipe as mp
from time import time
import random

# app = Flask(__name__)

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

# Setting landmarks for lefthand, righthand, leftleg, rightleg
bodyparts = ["left hand", "right hand", "left leg", "right leg"]
body_landmarks = {"left hand": 19, "right hand": 20, "left leg": 31, "right leg": 32}

def generate_random_coordinates(h, w):
    bodypart = random.randint(0, 3)
    if bodypart > 1:
        x = random.randint(50, w - 50)
        y = random.randint(h // 2, h - 50)
    else:
        x = random.randint(50, w - 50)
        y = random.randint(50, h - 50)
    return (bodyparts[bodypart], bodypart, x, y)


def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius = 2)
                                  )

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:

            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        

    # Otherwise
    #else:

        # Return the output image and the found landmarks.
    return output_image, landmarks
    
def generate_frames():
    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.
    video = cv2.VideoCapture(0)

    # Initialize a variable to store the time of the previous frame.
    time1 = 0
    touched = True
    points = 0

    coinImg = cv2.imread("COIN.png")
    coinImg = cv2.resize(coinImg, (50, 50), interpolation=cv2.INTER_AREA)

    time0 = time.time()

    while True:
        ok, frame = video.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        if touched:
            s, index, x, y = generate_random_coordinates(frame.shape[0], frame.shape[1])
            points += 1
            touched = False

        cv2.putText(frame, "Points: " + str(points), (500, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (72, 5, 51), 1.5)
        cv2.putText(frame, "Touch with: "+s, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (72, 5, 51), 1.5)
        
        height, width, channels = coinImg.shape
        offset = np.array((y-25, x-25))
        frame[offset[0]:offset[0]+50, offset[1]:offset[1]+50] = coinImg

        frame, landmarks = detectPose(frame, pose_video, display=False)
        
        if landmarks:
            x_bodypart = landmarks[body_landmarks[s]][0]
            y_bodypart = landmarks[body_landmarks[s]][1]
            
            if x-25 <= x_bodypart <= x+25 and y-25 <= y_bodypart <= y+25:
                touched = True

        time2 = time.time()

        if (time2 - time1) > 0:
            frames_per_second = 1.0 / (time2 - time1)
            # cv2.putText(frame, 'Time: '+str(round(time2-time0, 3)), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 165, 255), 1)

        time1 = time2
        cv2.putText(frame, 'Click esc to exit', (910, 620), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()
    cv2.destroyAllWindows()

@app.route('/videofeedrm')
def videofeedrm():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__=='__main__':
    app.run(debug=True)