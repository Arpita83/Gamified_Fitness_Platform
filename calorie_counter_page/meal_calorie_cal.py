import requests
import json

end_pt_url = " https://trackapi.nutritionix.com/v2/natural/nutrients"

query = {
 "query":"20 grams potato",
  "timezone": "US/Eastern"
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

for food in data['foods']:
    name = food['food_name']
    cal = food['nf_calories']
    serving_qty = food["serving_qty"]
    serving_unit = food["serving_unit"]

    #print(name, cal, duration)
    print("If your intake is " + str(serving_qty) +" "+ str(serving_unit) +" "+ str(name) +", Then the calories gained is " + str(cal))

