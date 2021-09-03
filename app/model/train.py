from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

#匯入檔案
emotion = pd.read_csv("./emotion.csv")

#編碼
label_map = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}
emotion['Class'] = emotion['type'].map(label_map)

#切割訓練集與資料集
X = emotion.drop(labels=['Class','type'],axis=1).values
y = emotion['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42) #8:2比例切分訓練集與資料集

#LDA降維
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis(n_components=3)
totX = LDA.fit_transform(X,y) #整個資料集
train_reduced = LDA.fit_transform(X_train,y_train) #訓練集
test_reduced = LDA.fit_transform(X_test,y_test) #測試集

#訓練(svm模型 / 核心=Polynomial)
#polyModel = svm.SVC(kernel='poly', degree=3, gamma="auto",C=1, max_iter=5, coef0=1.9, class_weight="balanced")
polyModel = svm.SVC(kernel="poly", C=0.01, gamma=0.5)
polyModel.fit(train_reduced, y_train)


''' 確認準確度
accuracy = polyModel.score(train_reduced, y_train)
acc = polyModel.score(test_reduced,y_test)

print(accuracy,acc)'''

#壓縮
import pickle
import gzip

with gzip.GzipFile('./svm-SER.pgz','w') as f:
    pickle.dump(polyModel, f)