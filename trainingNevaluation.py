import pandas as pd
import numpy as np

from sklearn import  linear_model
from sklearn.metrics import mean_squared_error

from math import sqrt

import matplotlib.pyplot as plt

train_set = pd.read_csv('train_set.csv')
test_set = pd.read_csv('test_set.csv')

X_train = np.array(train_set[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']])
y_train = np.array(train_set['sales_BEV'])

X_test = np.array(test_set[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']])
y_test = np.array(test_set['sales_BEV'])



lin_regr_perform = []
for i in range(10):
	lin_regr = linear_model.LinearRegression()
	lin_regr.fit(X_train, y_train)
	y_pred = lin_regr.predict(X_test)
	rmse = sqrt(mean_squared_error(y_test, y_pred))
	lin_regr_perform.append(rmse)

lin_regr_perform = np.array(lin_regr_perform)
avg_rmse_lin_regr = np.mean(lin_regr_perform)
print("lin_regr avg mean_squared_error: ",avg_rmse_lin_regr)

'''
ridge_perform = []
for i in range(10):
	regr_ridge = linear_model.Ridge(
		)
	regr_ridge.fit(X_train, y_train)
	y_pred_ridge = regr_ridge.predict(X_test)
	rmse_ridge = sqrt(mean_squared_error(y_test, y_pred_ridge))
	print(rmse_ridge)
	ridge_perform.append(rmse_ridge)

ridge_perform = np.array(ridge_perform)
avg_ridge = np.mean(ridge_perform)
print("ridge avg root mean_squared_error: ", avg_ridge)
'''




df_ita = pd.read_csv('it_EV.csv')
X_ita = np.array(df_ita[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']])
y_ita = np.array(df_ita[['sales_BEV']])
years = np.array(df_ita[['year']])



prediction_ita = lin_regr.predict(X_ita)
plt.plot(years,y_ita, 'o')
plt.plot(years, prediction_ita)
plt.xlabel("year")
plt.ylabel("Italy EV sales")
plt.show()
'''
prediction_ita_ridge = regr_ridge.predict(X_ita)
plt.plot(years,y_ita, 'o')
plt.plot(years, prediction_ita_ridge)
plt.xlabel("year")
plt.ylabel("Italy EV sales")
plt.title("Italy EV sales simulation")
plt.show()
'''