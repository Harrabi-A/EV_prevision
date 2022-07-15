import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv('EU_EV.csv')

print(df)
print(df.describe())
''' scaling
min_year= df['year'].min()
max_year = df['year'].max()
df['year'] = df['year'].apply(lambda x: (x-min_year)/(max_year-min_year))

min_fast_charging = df['fast_charging_point'].min()
max_fast_charging = df['fast_charging_point'].max()
avg_fast_charging = df['fast_charging_point'].mean()
df['fast_charging_point'] = df['fast_charging_point'].apply(lambda x: (x-avg_fast_charging)/(max_fast_charging-min_fast_charging))

min_slow_charging = df['slow_charging_point'].min()
max_slow_charging = df['slow_charging_point'].max()
avg_slow_charging = df['slow_charging_point'].mean()
df['slow_charging_point'] = df['slow_charging_point'].apply(lambda x: (x-avg_slow_charging)/(max_slow_charging-min_slow_charging))


min_sales_PHEV = df['sales_PHEV'].min()
max_sales_PHEV = df['sales_PHEV'].max()
avg_sales_PHEV = df['sales_PHEV'].mean()
df['sales_PHEV'] = df['sales_PHEV'].apply(lambda x: (x-avg_sales_PHEV)/(max_sales_PHEV-min_sales_PHEV))

min_sales_BEV = df['sales_BEV'].min()
max_sales_BEV = df['sales_BEV'].max()
avg_sales_BEV = df['sales_BEV'].mean()
df['sales_BEV'] = df['sales_BEV'].apply(lambda x: (x-avg_sales_BEV)/(max_sales_BEV-min_sales_BEV))

min_stock_BEV = df['stock_BEV'].min()
max_stock_BEV = df['stock_BEV'].max()
avg_stock_BEV = df['stock_BEV'].mean()
df['stock_BEV'] = df['stock_BEV'].apply(lambda x: (x-avg_stock_BEV)/(max_stock_BEV-min_stock_BEV))

min_stock_PHEV = df['stock_PHEV'].min()
max_stock_PHEV = df['stock_PHEV'].max()
avg_stock_PHEV = df['stock_PHEV'].mean()
df['stock_PHEV'] = df['stock_PHEV'].apply(lambda x: (x-avg_stock_PHEV)/(max_stock_PHEV-min_stock_PHEV))

min_EV_stock_share= df['EV_stock_share'].min()
max_EV_stock_share = df['EV_stock_share'].max()
avg_EV_stock_share = df['EV_stock_share'].mean()
df['EV_stock_share'] = df['EV_stock_share'].apply(lambda x: (x-avg_EV_stock_share)/(max_EV_stock_share-min_EV_stock_share))

min_EV_sales_share= df['EV_sales_share'].min()
max_EV_sales_share = df['EV_sales_share'].max()
avg_EV_sales_share = df['EV_sales_share'].mean()
df['EV_sales_share'] = df['EV_sales_share'].apply(lambda x: (x-avg_EV_sales_share)/(max_EV_sales_share-min_EV_sales_share))'''

#df['EV_stock_share'] = df['EV_stock_share'].apply(lambda x: x/100)
#df['EV_sales_share'] = df['EV_sales_share'].apply(lambda x: x/100)
print(df)

df1 = df.sample(frac = 1)

# reorder dataframe
dataset = df[['year','region','fast_charging_point','slow_charging_point','sales_BEV','sales_PHEV','EV_sales_share','stock_BEV','stock_PHEV','EV_stock_share']]
print(dataset)

Country = ['France','Belgium','Germany','Netherland','Poland','Portugal','Spain','Switzerland','United Kingdom','Denmark','Norway','Sweden','Finland']
Country_value = np.arange(13)

df = df[df['region'].isin(Country)]
df['region'].replace(Country, Country_value, inplace=True)

X = df[['year','fast_charging_point','slow_charging_point','stock_BEV','stock_PHEV','EV_stock_share']]
y = df['sales_BEV']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_set = X_train.assign(sales_BEV=y_train)
test_set = X_test.assign(sales_BEV=y_test)


# export dataset
train_set.to_csv('train_set.csv')

test_set.to_csv('test_set.csv')
