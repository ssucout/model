from flask import Flask, request, render_template

import json

from socket import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def receive_string():
    return 'fuck you  '



if __name__ == '__main__':
    app.run('0.0.0.0', port=80, debug=True)