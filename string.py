import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle



df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-9999, inplace=True)

print('Length is: ', int(math.ceil(0.1*len(df))))

forecast_out = int(math.ceil(0.1*len(df)))
#forecast_out = 2

print('1----------\n', df.head())
print('2----------\n', forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

print('3----------\n', df.head())

x = np.array(df.drop(['label'],1))
print('4----------\n', x)

x = preprocessing.scale(x)
print('5----------\n', x)

x_lately = x[-forecast_out:]
print('6----------\n', x_lately)

array = [5,4,6,8,7,2,3]
print('7----------\n', array)

newArray = array[:-2]
print('8----------\n', newArray)









