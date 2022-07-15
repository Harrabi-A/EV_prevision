import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer


from sklearn import  linear_model
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures


from math import sqrt

df = pd.read_csv('train_set.csv')

X = np.array(df[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']])
y = np.array(df['sales_BEV'])

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

for i in range(9):
	poly = PolynomialFeatures(degree=i)
	X_poly = poly.fit_transform(X_train)
	regr = linear_model.LinearRegression()
	regr.fit(X_poly, y_train)
	X_validation_poly = poly.fit_transform(X_validation)
	y_pred = regr.predict(X_validation_poly)
	rmse = sqrt(mean_squared_error(y_validation, y_pred))
	print("degree i=",i, " rmse=",rmse)

for i in range(9):
	regr_ridge = linear_model.Ridge(alpha=1e-1)
	regr_ridge.fit(X_train, y_train)
	y_pred_ridge = regr_ridge.predict(X_validation)
	rmse_ridge = sqrt(mean_squared_error(y_validation, y_pred_ridge))
	print("alpha i=",i, " rmse=",rmse_ridge)

def RMSE(y_true, y_pred):
	return sqrt(mean_squared_error(y_true, y_pred))

def scorer():
	return make_scorer(RMSE, greater_is_better=True)

ridge = linear_model.Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring=scorer(),cv=5)
ridge_regressor.fit(X_train,y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)