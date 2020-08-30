# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:09:22 2020

@author: Administrator
"""
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import uuid
import pandas as pd 
import datetime as dt
from datetime import datetime
import csv
import math
import pandas as pd
import matplotlib.dates as mpl_dates



data = pd.read_csv("Financial_Data/EurUsd.csv", names=["Date", "Open", "High","Low", "Close", "volume"])
#data = data[::-1]
data['volume'].astype('float')
data['label'].astype('str')
print(data.dtypes)
indexNames = data[ (data['volume'] < 999)].index
data.drop(indexNames , inplace=True)
data = data.reset_index(drop=True)
#data['time'] = pd.to_datetime(data['time'])
ohlc = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])

ohlc.set_index('Date', inplace = True) 

def CreateAndSaveCharts(data):
    i = 0;
    for x in range(len(data)):
         if x > 24:
             
             
             
           #  fig, axs = plt.subplots(3)
             df = data.iloc[x-24:x]
             df = df.reset_index(drop=True)
             #df.set_index('time', inplace = True) 
# Creating Subplots
'''
             axs[0].plot(df['time'],df['close'], linestyle='dashdot', color='blue', linewidth=7)
             axs[1].plot(df['time'],df['rsi'], linewidth=8, color='red')
             axs[2].plot(df['time'],df['open'],linestyle=':', color='green', linewidth=10)
            # axs[1].plot(kind='line',x='time',y='low',linestyle=':', color='orange')
             #df.plot(kind='line',x='time',y='low',linestyle=':', color='green', ax=ax)
        
             for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
                     '''
             if(data.iloc[x+4,4] < data.iloc[x, 4]):
                 fig.savefig('train/'+str(x)+'.png')
             else:
                 fig.savefig('train/'+str(x)+'.png')
             #xs.set_ylim([0,1])
             #plt.close()
             plt.show()
            # df.plot(kind='line',x='time',y='rsi', color='red', ax=ax)
        

CreateAndSaveCharts(data)
"""
#Below Code is code used to format and save the data

data['rsi'] = Rsi(data['close'])

#for x in data.index:
#    print(x)
 #   data.iloc[x,4] = data.iloc[x,4] - math.floor(data.iloc[x,4])


print(data['close'])
x = data[['rsi']].values.astype(float)
#z = data[['close']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)
#z_scaled = min_max_scaler.fit_transform(z)

df_normalized = pd.DataFrame(x_scaled)
#zdf_normalized = pd.DataFrame(z_scaled)
data['rsi'] = df_normalized 
#data['close'] = zdf_normalized
print(df_normalized)

OrganiseData(data)

Code Below used to set labels for the data 
for x in data.index:
    print(x)
    if data.iloc[x,4] < data.iloc[x+1,4]:
        data.iloc[x,6] = 'buy'
    else:
        data.iloc[x,6] = 'sell'
    
data_file = open('Financial_Data/EurUsdsxs.csv', 'w') 
data_file.write(data.to_csv(header =False, index=False, line_terminator='\n'))
data_file.close()
"""

