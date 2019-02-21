import datetime
import pandas as pd 
import matplotlib.pyplot as plt

start = datetime.datetime(2012,1,1)
end =   datetime.datetime(2017,1,1)

teslaDF = pd.read_csv('Tesla_Stock.csv',index_col= 'Date',parse_dates=True)
print(teslaDF.head())

fordDF = pd.read_csv('Ford_Stock.csv',index_col= 'Date',parse_dates=True)
print(fordDF.head())

GMDF = pd.read_csv('dataset/GM_Stock.csv',index_col= 'Date',parse_dates=True)
print(GMDF.head())


teslaDF['Open'].plot(label='Tesla',title='Opening Prices')

fordDF['Open'].plot(label='Ford')

GMDF['Open'].plot(label='GM')
plt.legend()


# volume of stock sold

teslaDF['Volume'].plot(label='Tesla',title='Volume Traded')

fordDF['Volume'].plot(label='Ford')

GMDF['Volume'].plot(label='GM')
plt.legend()


# Ford datapoints have huge spikes at late 2013
maxVal = fordDF['Volume'].max()
dateVal = fordDF['Volume'].argmax()
print(maxVal)
print(dateVal)

# Total traded 

teslaDF['Total Traded'] = teslaDF['Open']*teslaDF['Volume']
fordDF['Total Traded'] = fordDF['Open']*teslaDF['Volume']
GMDF['Total Traded'] = GMDF['Open']*GMDF['Volume']
'''
teslaDF['Total Traded'].plot(label= 'Tesla',figsize=(16,8))
fordDF['Total Traded'].plot(label='FORD')
GMDF['Total Traded'].plot(label='GM')
plt.legend()
'''

totaltradeMax = teslaDF['Total Traded'].max()
print(totaltradeMax)
totaltrademaxDate = teslaDF['Total Traded'].argmax()
print(totaltrademaxDate)

# Plot some rolling avg

GMDF['MA50'] =  GMDF['Open'].rolling(window=50).mean()
GMDF['MA200'] =  GMDF['Open'].rolling(window=200).mean()
GMDF[['Open','MA50','MA200']].plot()

# Relationships between different company stocks 

from pandas.plotting import scatter_matrix

car_companiesDF = pd.concat([teslaDF['Open'],fordDF['Open'],GMDF['Open']],axis=1)
print(car_companiesDF.head())

car_companiesDF.columns =  ['Tesla Open','Ford Open','GM Open']
print(car_companiesDF.head())

scatter_matrix(car_companiesDF,alpha=0.2,hist_kwds={'bins':50})










































