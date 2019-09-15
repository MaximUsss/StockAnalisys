from stocks_analysis_models import train_model

import math
import datetime
import pandas_datareader.data as web
import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 1, 12)

df = web.DataReader("AAPL", 'yahoo', start, end)

dfreg = df.loc[:, ['Adj Close', 'Volume']]
# High Low Percentage
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
# Percentage Change
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
# math.ceil - round to nearest high value
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
# shift values in "label" column by 1 to form training data
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
# excludes column "label" (predicted value) to form a training and test set
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
# https://scikit-learn.org/stable/modules/preprocessing.html
X = preprocessing.scale(X)

# Finally, we want to find Data Series of late X and early X (train) for model generation and evaluation
# a[start:stop:step]
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[: -forecast_out]
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = train_model("linear", X_train, y_train, X_test, y_test)
forecast_set = clf.predict(X_lately)
dfreg['Forecast'] = np.nan

# print the results
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
