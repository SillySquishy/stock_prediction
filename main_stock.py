import pandas as pd
import numpy as np
import datetime, math
import pandas_datareader.data as web
from pandas import Series, DataFrame

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

style.use("ggplot")

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2018,1,11)

df = web.DataReader("AAPL","yahoo",start,end)
print(df.tail())


# this will find out the  moving average for the last 100 days (100 windows)
dfreg = df.loc[:,["Adj Close","Volume"]]
dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)


#print(dfreg.head())

###  givning the features ###
x = np.array(dfreg.drop(["label"],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out:]
dfreg.dropna(inplace=True)

### labels ###
y = np.array(dfreg['label'])
y = np.array(dfreg['label'])

#print(len(x), len(y))

### creating training and testing data ###
# 20% of data will be testing data
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2)

### classifier ###
# using 10 threads
clf = LinearRegression(n_jobs=10)
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)

#print(accuracy)

forecast_set = clf.predict(x_lately)

print(forecast_set,accuracy,forecast_out)
dfreg["forecast"] = np.nan

last_date = dfreg.iloc[-1].name

last_unix = last_date.timestamp()

one_day = 86400

next_unix = last_unix + one_day

# added the forecast to the tail of the dataframe
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg["Adj Close"].tail(500).plot()
dfreg["forecast"].tail(500).plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()