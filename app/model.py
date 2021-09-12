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
