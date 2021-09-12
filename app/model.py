# -*- coding: utf-8 -*-

import gzip
import pickle
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#載入model
with gzip.open('app/model/svm-SER.pgz','r') as f:
    polyModel = pickle.load(f)

def predict(angryEmo, happyEmo, neutralEmo, sadEmo):
    #以喜怒哀中四種情緒的標籤降維後下去進行預測
    angryTag = polyModel.predict([angryEmo[-1]])[0]
    happyTag = polyModel.predict([happyEmo[-1]])[0]
    neutralTag = polyModel.predict([neutralEmo[-1]])[0]
    sadTag = polyModel.predict([sadEmo[-1]])[0]

    #以出現最多的情緒做結
    predictEmoQuantity = [angryTag,happyTag,neutralTag,sadTag]
    emoType = max(predictEmoQuantity,key = predictEmoQuantity.count)

    print(emoType)
    return emoType

def dimensionalityReduction(emoFile):
    LDA = LinearDiscriminantAnalysis(n_components=3)

    #匯入基準檔案
    emotion = pd.read_csv("./emotion.csv")

    #編碼
    label_map = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}
    emotion['Class'] = emotion['type'].map(label_map)

    #切分特徵與類別
    X = emotion.drop(labels=['Class','type'],axis=1).values
    y = emotion['Class'].values

    #將新的資料特徵加入基準特徵中
    newFeature = emoFile.values[0][1:386]
    newX = np.vstack((X,newFeature))

    #給予新的資料特徵四種不同情緒的標籤
    angryY = np.append(y,0)
    happyY = np.append(y,1)
    neutralY = np.append(y,2)
    sadY = np.append(y,3)

    #以四種不同情緒的標籤進行降維
    angryEmo = LDA.fit_transform(newX,angryY)
    happyEmo = LDA.fit_transform(newX,happyY)
    neutralEmo = LDA.fit_transform(newX,neutralY)
    sadEmo = LDA.fit_transform(newX,sadY)

    return angryEmo,happyEmo,neutralEmo,sadEmo
