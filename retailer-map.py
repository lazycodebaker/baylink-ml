import requests
import json

API_URL = "https://webapi.fieldproxy.com/v3/zapier/sheets?sheet_id=retailer&per_page=1000&page=1&startDate=2024-02-21&endDate=2024-06-24"
API_KEY = "U2FsdGVkX19G5lwpp3Fpg/ce90tfXZKBdmr2BpSeyhbARerGpbJLakdT0+J66mQT6Z9Kp8eXBZZ61kKhI0r1ge+mBO/p/YrtHXEdVLHg7Jbz9FDvymEXdTmKBEAiWZBW"
headers = {"x-api-key": API_KEY}
response = requests.get(API_URL, headers=headers)

if response.status_code == 200:
    data = response.json()
    parsed_json = json.dumps(data, indent=2)
    
    for retailer in data:
        print("Retailer: ", retailer.get('name'))
        # for key, value in retailer.items():
        #     if value == None:
        #         val = input(f"Enter the value for key: {key} \n")
        #         retailer[key] = val
    with open('example.json', 'w') as file:
        json.dump(data, file, indent=4)
        
else:
    print(f"Error: {response.status_code}")
