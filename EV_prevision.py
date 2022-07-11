import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import  linear_model

from math import sqrt

df = pd.read_csv('it_EV.csv')
print (df)

X = np.array(df[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']])
#X = np.array(df[['year','fast_charging_point','slow_charging_point','stock_BEV']])
y = np.array(df['sales_BEV'])
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#regr = linear_model.LinearRegression()
regr = linear_model.Ridge(alpha=5.)

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
print("mean_squared_error: ",rmse)

