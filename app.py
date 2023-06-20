### Integrate HTML With Flask
### HTTP verb GET And POST
from flask import Flask,redirect,url_for,render_template,request

app=Flask(__name__)

# @app.route('/')
# def welcome():
#     return render_template('index_f.html')

# @app.route('/index.html')
# def index():
#     return render_template('index_f.html')

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
    return render_template("index_f.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__=='__main__':
    app.run(debug=True)
