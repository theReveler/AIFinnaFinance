import math
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from StockPredictor import ParseData

def KNNRegression(NNeighbors, D, interval=1):
    knnIn = np.zeros((len(D.Timestamp),1))
    knnTarget = np.zeros((len(D.Timestamp),1))
    for i in range (0, len(D.Timestamp)):
        knnIn[i] = D.Timestamp[i]
        knnTarget[i] = D.close[i]
    knnPredict = np.zeros((len(D.Timestamp),1))
    knnPredict[0:NNeighbors+1] = 0
    R = KNeighborsRegressor(NNeighbors, 'distance')
    for i in range(NNeighbors+1,len(D.Timestamp)):
        R.fit(knnIn[(i-(NNeighbors+1)):i-1].reshape(-1, 1), knnTarget[(i-(NNeighbors+1)):i-1])
        knnPredict[i] = R.predict(knnIn[i].reshape(-1, 1))
    Err = 0
    for i in range(NNeighbors+1, len(D.Timestamp)):
        Err += math.sqrt(((D.close[i] - knnPredict[i])/D.close[i])**2)
    Err = Err / (len(D.Timestamp) - NNeighbors)
    print(Err)


def KNNClassification(NNeighbors, D, interval=1):
    knnIn = np.zeros((len(D.Timestamp),1))
    knnTarget = np.zeros((len(D.Timestamp),1))
    for i in range (interval, len(D.Timestamp),interval):
        knnIn[i] = D.Timestamp[i]
        if(D.close[i] - D.close[i-interval] > 0):
            knnTarget[i] = 1
        else:
            knnTarget[i] = 0
    knnPredict = np.zeros((len(D.Timestamp),1))
    knnPredict[0:NNeighbors*interval+1] = 0
    R = KNeighborsClassifier(NNeighbors)
    for i in range((NNeighbors+1)*interval, len(D.Timestamp), interval):
        R.fit(knnIn[range(i - ((NNeighbors)*interval),i, interval)].reshape(-1, 1), knnTarget[range(i - ((NNeighbors)*interval),i,interval)].ravel())
        knnPredict[i] = R.predict(knnIn[i].reshape(-1, 1))
        print(knnIn[range(i - ((NNeighbors)*interval),i, interval)].reshape(-1, 1), knnTarget[range(i - ((NNeighbors)*interval),i,interval)].ravel(), knnTarget[i], knnPredict[i])
    Err = 0
    for i in range((NNeighbors+1)*interval, len(D.Timestamp),interval):
        if knnPredict[i] != knnTarget[i]:
            Err += 1
        #print(knnTarget[i], knnPredict[i], i)
    Err = Err / ((len(D.Timestamp) - NNeighbors)/interval)
    print(Err)
path = os.path.dirname(os.path.realpath(__file__)) + '\\weekly_IXIC.csv'
D = ParseData(path)
KNNClassification(7, D,4)
