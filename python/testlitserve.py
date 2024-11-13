import requests
import io
import base64


URL = "http://localhost:8000/predict"

path="./AE.png"

def get_prediction(path:str):

    # open file
    with open(path, 'rb') as f:
        imagebytes = f.read()

    # base64 encode
    encoded_data = base64.b64encode(imagebytes)

    # to uft-8
    data = {"image_data": encoded_data.decode('utf-8')}

    # send post request
    headers = {'Content-Type': 'application/json'}
    response = requests.post(URL, json=data, headers=headers)
    
    return response

response = get_prediction(path)

print(response.status_code)  # 200 OK
print(response.json())  # {'result': 'success'}