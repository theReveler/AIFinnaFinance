import os
import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
from StockPredictor import ParseData
from StockPredictor import PlotData

def plotScaledData(filePath):
	pth = filePath + '\\daily_adjusted_AMZN.csv'
	A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 3))
	A = scale(A)
	#y is the dependent variable
	y = A[:, 1].reshape(-1, 1)
	#A contains the independent variable
	A = A[:, 0].reshape(-1, 1)
	#Plot the high value of the stock price
	mpl.plot(A[:, 0], y[:, 0])
	mpl.show()
	return
	
if __name__ == "__main__":
	path = os.path.dirname(os.path.realpath(__file__))
	#plotScaledData(path)
	df = ParseData(path + '\\validation.csv')
	PlotData(df)