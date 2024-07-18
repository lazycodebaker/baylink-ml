import requests

# Define the API endpoint
api_url = "https://backend.baylink.in/api/retailers/all"

# Make a GET request to the API
response = requests.get(api_url)


# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()
    print(data)
else:
    print(f"Failed to retrieve data: {response.status_code}")
