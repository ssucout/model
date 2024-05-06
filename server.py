from flask import Flask, request, render_template
from flask import send_file
import base64
import json

from socket import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def receive_string():
    with open("image.png", "rb") as image_file:
        image_binary = image_file.read()
        encoded_string = base64.b64encode(image_binary)

        image_dict = {
            "image.png": encoded_string.decode()
        }

        image_json = json.dumps(image_dict)

        print(image_json)
    


if __name__ == '__main__':
    app.run('0.0.0.0', port=80, debug=True)