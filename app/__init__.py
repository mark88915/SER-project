# -*- coding: utf-8 -*-

import numpy as np
import app.model as model

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def postInput():
    
    '''insertValue = request.get_json()
    f1 = insertValue['feature1']
    f2 = insertValue['feature2']
    f3 = insertValue['feature3']
    input = np.array([[f1, f2, f3]])

    emoType = model.predict(input)

    return jsonify({'return':str(emoType)})'''

    file = request.files['file']

    if file.filename == "":
        return "no file"
    else:
        print(file.filename)

    return "Success123~"
