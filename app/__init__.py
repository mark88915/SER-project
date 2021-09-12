# -*- coding: utf-8 -*-

import numpy as np
import app.model as model
import pandas as pd

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

    #請求檔案
    file = request.files['file']

    #確認是否有檔案
    if file.filename == "":
        return "no file"
    else:
        print(file.filename)

    #讀取csv
    emoFile = pd.read_csv(file)
    #print(emoFile)

    #進行降維
    angryEmo, happyEmo, neutralEmo, sadEmo = model.dimensionalityReduction(emoFile)

    #預測
    emoType = model.predict(angryEmo, happyEmo, neutralEmo, sadEmo)

    '''if(emoType == 0):
        return "怒"
    elif(emoType == 1):
        return "喜"
    elif(emoType == 2):
        return "中"
    elif(emoType == 3):
        return "哀"'''

    print(emoType)

    return "Success"
