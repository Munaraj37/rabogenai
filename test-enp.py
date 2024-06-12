import requests

url = "https://southeastasia.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1"#replace southeastasia whith your Speech resouce region
subscription_key = "37d5f229e5bc4a1db5b342f0601b3ea6"

headers = {
    "Ocp-Apim-Subscription-Key": subscription_key
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    ice_info = response.json()
    print("ICE Information:", ice_info)
else:
    print(f"Failed to fetch ICE information. Status code: {response.status_code}")
    print("Response:", response.text)
