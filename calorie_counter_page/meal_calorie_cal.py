import requests
import json

end_pt_url = " https://trackapi.nutritionix.com/v2/natural/nutrients"

food_item = input("Enter food and its amount (Ex. 1 cup veg cup soup): ")

query = {
 "query":food_item,
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

for food in data['foods']:
    name = food['food_name']
    cal = food['nf_calories']
    serving_qty = food["serving_qty"]
    serving_unit = food["serving_unit"]

    #print(name, cal, duration)
    print("If your intake is " + str(serving_qty) +" "+ str(serving_unit) +" "+ str(name) +", Then the calories gained is " + str(cal))

