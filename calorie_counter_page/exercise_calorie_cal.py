import requests
import json

end_pt_url = " https://trackapi.nutritionix.com/v2/natural/exercise"

query = {
 "query":"30 minutes aerobics",
 "gender":"female",
 "weight_kg":72.5,
 "height_cm":167.64,
 "age":30
}
api_id = "66e88fcd"
api_key = "915bd34e3996d68e870d3be75c07b467"

HEADERS = {"x-app-id": api_id,
"x-app-key": api_key,
"Content-Type": "application/json"}
r = requests.post(end_pt_url, headers=HEADERS,json=query)
data = json.loads(r.text)
new_string = json.dumps(data, indent = 2)
#print(new_string)

for exercise in data['exercises']:
    name = exercise['name']
    cal = exercise['nf_calories']
    duration = exercise["duration_min"]
    #print(name, cal, duration)
    print("If the activity is " + str(name) + " and the duration is " + str(duration) + ", Then the calories burnt is " + str(cal))