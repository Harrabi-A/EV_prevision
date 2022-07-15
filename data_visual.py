import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("EV.csv")

df = df[df['region'].isin(['Italy','France','Germany','Norway','Sweden'])]
# Select cars data
df_car = df[df['mode'].isin(['Cars','EV'])]     
# Select historical data
df_car = df_car[df_car['category'].isin(['Historical'])]
# Setting parameter as an index
df_car.set_index('parameter', inplace=True)
# Drop colom region, mode, category


df_it_car = df_car[df_car['region'].isin(['Italy'])]
df_fr_car = df_car[df_car['region'].isin(['France'])]
df_de_car = df_car[df_car['region'].isin(['Germany'])]
df_nw_car = df_car[df_car['region'].isin(['Norway'])]
df_sw_car = df_car[df_car['region'].isin(['Sweden'])]

df_car = df_car[['value','powertrain','year']]
df_it_car = df_it_car[['value','powertrain','year']]
df_fr_car = df_fr_car[['value','powertrain','year']]
df_de_car = df_de_car[['value','powertrain','year']]
df_nw_car = df_nw_car[['value','powertrain','year']]
df_sw_car = df_sw_car[['value','powertrain','year']]

# get a list of dataframe index
feature = df_it_car.index.unique()


dataset = pd.pivot_table(df_car, values='value', index=['year'], columns=['parameter','powertrain'])
dataset_it = pd.pivot_table(df_it_car, values='value', index=['year'], columns=['parameter','powertrain'])
dataset_fr = pd.pivot_table(df_fr_car, values='value', index=['year'], columns=['parameter','powertrain'])
dataset_de = pd.pivot_table(df_de_car, values='value', index=['year'], columns=['parameter','powertrain'])
dataset_nw = pd.pivot_table(df_nw_car, values='value', index=['year'], columns=['parameter','powertrain'])
dataset_sw = pd.pivot_table(df_sw_car, values='value', index=['year'], columns=['parameter','powertrain'])


# Rename column
dataset.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']
dataset_it.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']
dataset_fr.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']
dataset_de.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']
dataset_nw.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']
dataset_sw.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']

#compute mean
dataset = dataset.fillna(value=0)
dataset_mean = dataset.groupby(['year']).mean()
print(dataset)


# Data Visualization
years = np.array(dataset_it.index)

# SALES
BEV_sales_mean = np.array(dataset_mean['sales_BEV'])
plt.plot(years, BEV_sales_mean, label='mean', linestyle='dotted', linewidth='2')
BEV_sales_it = np.array(dataset_it['sales_BEV'])
plt.plot(years, BEV_sales_it, label='Italy', linewidth='0.7')
BEV_sales_fr = np.array(dataset_fr['sales_BEV'])
plt.plot(years, BEV_sales_fr, label='France',linewidth='0.7')
BEV_sales_de = np.array(dataset_de['sales_BEV'])
plt.plot(years, BEV_sales_de, label='Germany',linewidth='0.7')
BEV_sales_nw = np.array(dataset_nw['sales_BEV'])
plt.plot(years, BEV_sales_nw, label='Norway', linewidth='0.7')
BEV_sales_sw = np.array(dataset_sw['sales_BEV'])
plt.plot(years, BEV_sales_sw, label='Sweden', linewidth='0.7')

plt.legend(loc="upper left")
plt.ylabel("BEV sales")
plt.xlabel("year")
plt.show()



# FAST CHARGING POINT
fast_charging_point_mean = np.array(dataset_mean['fast_charging_point'])
plt.plot(years, fast_charging_point_mean, label='mean', linestyle='dotted', linewidth='2')
fast_charging_point_it = np.array(dataset_it['fast_charging_point'])
plt.plot(years, fast_charging_point_it, label='Italy', linewidth='0.7')
fast_charging_point_fr = np.array(dataset_fr['fast_charging_point'])
plt.plot(years, fast_charging_point_fr, label='France',linewidth='0.7')
fast_charging_point_de = np.array(dataset_de['fast_charging_point'])
plt.plot(years, fast_charging_point_de, label='Germany',linewidth='0.7')
fast_charging_point_nw = np.array(dataset_nw['fast_charging_point'])
plt.plot(years, fast_charging_point_nw, label='Norway', linewidth='0.7')
fast_charging_point_sw = np.array(dataset_sw['fast_charging_point'])
plt.plot(years, fast_charging_point_sw, label='Sweden', linewidth='0.7')

plt.legend(loc="upper left")
plt.ylabel("fast_charging_point")
plt.xlabel("year")
plt.show()

# EV SALES SHARE
EV_sales_share_mean = np.array(dataset_mean['EV_sales_share'])
plt.plot(years, EV_sales_share_mean, label='mean', linestyle='dotted', linewidth='2')
EV_sales_share_it = np.array(dataset_it['EV_sales_share'])
plt.plot(years, EV_sales_share_it, label='Italy', linewidth='0.7')
EV_sales_share_fr = np.array(dataset_fr['EV_sales_share'])
plt.plot(years, EV_sales_share_fr, label='France',linewidth='0.7')
EV_sales_share_de = np.array(dataset_de['EV_sales_share'])
plt.plot(years, EV_sales_share_de, label='Germany',linewidth='0.7')
EV_sales_share_nw = np.array(dataset_nw['EV_sales_share'])
plt.plot(years, EV_sales_share_nw, label='Norway', linewidth='0.7')
EV_sales_share_sw = np.array(dataset_sw['EV_sales_share'])
plt.plot(years, EV_sales_share_sw, label='Sweden', linewidth='0.7')

plt.legend(loc="upper left")
plt.ylabel("EV_sales_share")
plt.xlabel("year")
plt.show()

#Save data for next step
dataset.to_csv("clean_EV.csv")
