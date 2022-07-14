import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model, ensemble

from sklearn.metrics import mean_squared_error

from math import sqrt

df = pd.read_csv('train_set.csv')


X = np.array(df[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']])
y = np.array(df['sales_BEV'])


# Linear Regression
lin_regr = linear_model.LinearRegression()
lin_regr_perform = []
# Gradient Boosting
grad_boost = ensemble.GradientBoostingRegressor()
grad_boost_perform = []
# Adaptive Boosting
ada_boost = ensemble.AdaBoostRegressor()
ada_boost_perform = []

kf = KFold(n_splits=10, random_state=None, shuffle=False)
for train_index, validation_index in kf.split(X):
	#print("TRAIN:", train_index, "VALIDATION:", validation_index)
	X_train = X[train_index[0]:train_index[-1]]
	X_validation = X[validation_index[0]:validation_index[-1]]
	y_train= y[train_index[0]:train_index[-1]]
	y_validation = y[validation_index[0]:validation_index[-1]]
	lin_regr.fit(X_train, y_train)
	grad_boost.fit(X_train, y_train)
	ada_boost.fit(X_train, y_train)
	lin_regr_ypred = lin_regr.predict(X_validation)
	grad_boost_ypred = grad_boost.predict(X_validation)
	ada_boost_ypred = ada_boost.predict(X_validation)
	mse_lin_regr = mean_squared_error(y_validation, lin_regr_ypred)
	mse_grad_boost = mean_squared_error(y_validation, grad_boost_ypred)
	mse_ada_boost =  mean_squared_error(y_validation, ada_boost_ypred)
	#print("mse_lin_regr: ", mse_lin_regr)
	#print("grad_boost ", mse_grad_boost)
	#print("ada_boost: ", mse_ada_boost)
	lin_regr_perform.append(sqrt(mse_lin_regr))
	grad_boost_perform.append(sqrt(mse_grad_boost))
	ada_boost_perform.append(sqrt(mse_ada_boost))

lin_regr_perform = np.array(lin_regr_perform)
grad_boost_perform = np.array(grad_boost_perform)
ada_boost_perform = np.array(ada_boost_perform)

mean_mse_lin_regr = np.mean(lin_regr_perform)
mean_mse_grad_boost = np.mean(grad_boost_perform)
mean_mse_ada_boost = np.mean( ada_boost_perform)

print("mean_rmse_lin_regr: ", mean_mse_lin_regr)
print("mean_rmse_grad_boost: ", mean_mse_grad_boost)
print("mean_rmse_ada_boost:", mean_mse_ada_boost)
