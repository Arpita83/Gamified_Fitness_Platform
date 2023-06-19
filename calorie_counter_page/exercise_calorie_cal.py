import requests
import json

end_pt_url = " https://trackapi.nutritionix.com/v2/natural/exercise"

Exercise = input("Enter exercise with duration/count (Ex. Running 3 miles): ")
Gender = input("Enter gender: ") #female
Wt = int(input("Enter weight in kgs: ")) #72.5
Ht = int(input("Enter Height in cms: ")) # 167.64
Age = int(input("Enter age: " )) # 20

query = {
 "query":Exercise,
 "gender":Gender,
 "weight_kg":Wt,
 "height_cm":Ht,
 "age":Age
}
api_id = "API ID"
api_key = "API KEY"

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
