# -*- coding: utf-8 -*-

import gzip
import pickle

#載入model
with gzip.open('app/model/svm-SER.pgz','r') as f:
    polyModel = pickle.load(f)

def predict(input):
    pred = polyModel.predict(input)[0]
    print(pred)
    return pred