import math
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from StockPredictor import ParseData

def KNNRegression(NNeighbors, D, interval=1):
    #Create arrays for features (timestamp) and targets (closing price)
    knnIn = np.zeros((len(D.Timestamp),1))
    knnTarget = np.zeros((len(D.Timestamp),1))
    #Populate arrays with data from the input matrix
    for i in range (0, len(D.Timestamp)):
        knnIn[i] = D.Timestamp[i]
        knnTarget[i] = D.close[i]
    #Create the prediction array (What will be tested against the closing price of a day)
    knnPredict = np.zeros((len(D.Timestamp),1))
    #Don't have the data to predict for the first NNeighbors days (because prediction requires at least NNeighbors previous points)
    knnPredict[0:NNeighbors+1] = 0
    #Create the regressor using a distance-weightedd KNN function
    R = KNeighborsRegressor(NNeighbors, 'distance')
    for i in range(NNeighbors+1,len(D.Timestamp)):
        #Fit the regressor to the current window (past NNeighbor points and their closing value targets)
        R.fit(knnIn[(i-(NNeighbors+1)):i-1].reshape(-1, 1), knnTarget[(i-(NNeighbors+1)):i-1])
        #Make a prediction for the current timestamp
        knnPredict[i] = R.predict(knnIn[i].reshape(-1, 1))
    Err = 0
    for i in range(NNeighbors+1, len(D.Timestamp)):
        #Compute the total mean absolute error percentage
        Err += math.sqrt(((D.close[i] - knnPredict[i])/D.close[i])**2)
    #Divide summed mean absolute error percentage by number of points considered 
    Err = Err / (len(D.Timestamp) - NNeighbors)
    print(Err)

def KNNClassification(NNeighbors, D, interval=1):
    #Create arrays for features (timestamp) and targets (positive/negative return)
    knnIn = np.zeros((len(D.Timestamp),1))
    knnTarget = np.zeros((len(D.Timestamp),1))
    for i in range (interval, len(D.Timestamp),interval):
        #Populate input variable with time data
        knnIn[i] = D.Timestamp[i]
        #Compute positive/negative return since last interval
        if(D.close[i] - D.close[i-interval] > 0):
            knnTarget[i] = 1
        else:
            knnTarget[i] = 0
    #Create prediction array for storing predicted values 
    knnPredict = np.zeros((len(D.Timestamp),1))
    knnPredict[0:NNeighbors*interval+1] = 0
    #Create the KNN classifier using NNeighbors as a parameter
    R = KNeighborsClassifier(NNeighbors)
    for i in range((NNeighbors+1)*interval, len(D.Timestamp), interval):
        #Fit the classifier to the NNeighbor points based on the interval and the target return
        R.fit(knnIn[range(i - ((NNeighbors)*interval),i, interval)].reshape(-1, 1), knnTarget[range(i - ((NNeighbors)*interval),i,interval)].ravel())
        #Make a prediction with the fitted classifier
        knnPredict[i] = R.predict(knnIn[i].reshape(-1, 1))
        #DEBUG
        #print(knnIn[range(i - ((NNeighbors)*interval),i, interval)].reshape(-1, 1), knnTarget[range(i - ((NNeighbors)*interval),i,interval)].ravel(), knnTarget[i], knnPredict[i])
    Err = 0
    #Compute the classification accuracy
    for i in range((NNeighbors+1)*interval, len(D.Timestamp),interval):
        if knnPredict[i] != knnTarget[i]:
            Err += 1
        #print(knnTarget[i], knnPredict[i], i)
    #Divide the calculated error based on number of samples (input matrices could be sparse based on input)
    Err = Err / ((len(D.Timestamp) - NNeighbors)/interval)
    print(Err)

path = os.path.dirname(os.path.realpath(__file__)) + '\\daily_adjusted_AMZN.csv'
D = ParseData(path)

for i in range(1,13,2):
    print(i)
    KNNRegression(i,D)