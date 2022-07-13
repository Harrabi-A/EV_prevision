import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("EV.csv")

Country = ['France','Belgium','Germany','Italy','Denmark','Norway','Sweden','Finland','Netherland','Poland','Portugal','Spain','Switzerland','United Kingdom']
#debug 
#print(df)

#for i in Country:
# Select region data data
df = df[df['region'].isin(Country)]
#print(df)
    # Select cars data
df_EV = df[df
['mode'].isin(['Cars','EV'])]     
    #print(df_EV)
    # Select historical data

df_EV = df_EV[df_EV['category'].isin(['Historical'])]
    #print(df_EV)
df_EV = df_EV[df_EV['parameter'].isin(['EV charging points','EV sales','EV sales share','EV stock','EV stock share'])]
    # Setting parameter as an index
df_EV.set_index('parameter', inplace=True)
    # Drop colom mode, category
df_EV = df_EV[['value','powertrain','year','region']]
#print(df_EV)
    # get a list of dataframe index
feature = df_EV.index.unique()
    #print(feature)

    
#print(df_EV)

dataset = pd.pivot_table(df_EV, values='value', index=['year','region'], columns=['parameter','powertrain'])
#print(dataset)

    # Rename column
dataset.columns = ['fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']
    # Filling missing data with zeros (best choice)

dataset = dataset.fillna(value=0)
print(dataset)

    # export Dataframe to csv file
dataset.to_csv('EU_EV.csv')


