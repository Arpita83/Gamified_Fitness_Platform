### Integrate HTML With Flask
### HTTP verb GET And POST
from flask import Flask,redirect,url_for,render_template,request

app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

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

@app.route('/meet_the_team.html')
def team():
    return render_template('meet_the_team.html')

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
    api_id = "66e88fcd"
    api_key = "915bd34e3996d68e870d3be75c07b467"

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

    api_id = "66e88fcd"
    api_key = "915bd34e3996d68e870d3be75c07b467"

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

if __name__=='__main__':
    app.run(debug=True)