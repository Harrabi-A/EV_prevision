import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("EV.csv")

#debug 
#print(df)

# Select Italy data
df_it = df[df['region'].isin(['Italy'])]
#print(df_it)
# Select cars data
df_it_car = df_it[df_it['mode'].isin(['Cars','EV'])]     
#print(df_it_car)
# Select historical data
df_it_car = df_it_car[df_it_car['category'].isin(['Historical'])]
#print(df_it_car)
# Setting parameter as an index
df_it_car.set_index('parameter', inplace=True)
# Drop colom region, mode, category
df_it_car = df_it_car[['value','powertrain','year']]
#print(df_it_car)
# get a list of dataframe index
feature = df_it_car.index.unique()
#print(feature)

dataset = pd.pivot_table(df_it_car, values='value', index=['year'], columns=['parameter','powertrain'])

# Rename column
dataset.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']
# Filling missing data with zeros (best choice)
dataset = dataset.fillna(value=0)
print(dataset)

# export Dataframe to csv file
#dataset.to_csv('it_EV.csv')

# Data Visualization
'''years = np.array(dataset.index)
BEV_sales = np.array(dataset['sales_BEV'])
plt.plot(years, BEV_sales)
plt.ylabel("BEV sales")
plt.show()'''

# Save Dtaset in csv file 