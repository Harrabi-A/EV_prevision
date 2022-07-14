import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn import ensemble

from math import sqrt

df = pd.read_csv('train_set.csv')

X = np.array(df[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']])
y = np.array(df['sales_BEV'])


X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.15)

# Gradient Boosting
grad_boost = ensemble.GradientBoostingRegressor()

grid = {
	'learning_rate':[0.01,0.05,0.1],
	'n_estimators':np.arange(100,500,1000),
	'min_samples_split': [2,5,10],
	'max_depth': [3,4,5,6]
}

def RMSE(y_true, y_pred):
	return sqrt(mean_squared_error(y_true, y_pred))

def scorer():
	return make_scorer(RMSE, greater_is_better=True)


grad_boost_cv = GridSearchCV(grad_boost, grid, cv = 10, scoring=scorer())
grad_boost_cv.fit(X_train, y_train)

print("Best Parameters:", grad_boost_cv.best_params_)
print("Train Score:", grad_boost_cv.best_score_)

gb_tuned= ensemble.GradientBoostingRegressor(**grad_boost_cv.best_params_)
gb_tuned.fit(X_train, y_train)
grad_boost.fit(X_train, y_train)
deafault_grad_boost_pred = grad_boost.predict(X_validation)
fine_tuned_grad_boost_pred = gb_tuned.predict(X_validation)
tuned_rmse = sqrt(mean_squared_error(y_validation, fine_tuned_grad_boost_pred))
default_rmse = sqrt(mean_squared_error(y_validation, deafault_grad_boost_pred))
print("default gradient Boost rmse: ", default_rmse)
print("fine tuned gradient Boost rmse: ", tuned_rmse)




