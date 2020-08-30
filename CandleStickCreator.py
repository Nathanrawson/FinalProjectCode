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
import gc

import mplfinance as mpf

df = pd.read_csv("Financial_Data/EurUsd.csv", names=["Date", "Open", "High","Low", "Close", "volume"])
indexNames = df[ (df['volume'] < 999)].index
df.drop(indexNames , inplace=True)
df = df.reset_index(drop=True)
ohlc = df.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc.set_index('Date', inplace = True) 



#Below RSI Function created by FutureSharks at the link below 
#https://github.com/FutureSharks/ml-finance/blob/master/examples/8-deep-q-forex/rsi.py


def CreateAndSaveCharts(ohlc):
    tenRowDataArrays = []
    i = 0;
    
    for x in range(len(ohlc)):
         if x > 24:
             data = ohlc.iloc[x-24:x]
             
             fig = mpf.figure(style='charles',figsize=(7,8))
             ax1 = fig.add_subplot(2,1,1) 
             plt.ioff()
             print(x)
 
             if(ohlc.iloc[x+2,3] < ohlc.iloc[x, 3]):
                 #save = dict(fname='train2/sell/'+str(x)+'.png',dpi=30,pad_inches=0.25)
                 mpf.plot(data, ax = ax1, type='candle',style='charles', mav=(3,6,9))
                 ax1.set_xticks([])
                 ax1.set_yticks([])
                 fig.savefig('train2/sell/'+str(x)+'.png')
             else:
                 #save = dict(fname='train2/buy/'+str(x)+'.png',dpi=30,pad_inches=0.25)
                 mpf.plot(data,  ax = ax1,type='candle',style='charles', mav=(3,6,9), axisoff=True)
                 ax1.set_xticks([])
                 ax1.set_yticks([])
                 fig.savefig('train2/buy/'+str(x)+'.png')
                 
             fig.clf()
             fig.clear()
             plt.close(fig)
             ax1.clear()
             gc.collect()
     
             del(fig)
             del(ax1)
             del(data)
        
CreateAndSaveCharts(ohlc)

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

