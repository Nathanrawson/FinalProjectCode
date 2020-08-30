#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:06:29 2020

@author: nathan
"""

from numpy import genfromtxt
import matplotlib.pyplot as plt
import mpl_finance
import numpy as np
import uuid
import requests
import pandas as pd
import json
import csv
from datetime import datetime
from pandas.io.json import json_normalize
from datetime import timezone

dt = datetime(2020, 4, 2)
timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
response = requests.get("https://api.kraken.com/0/public/OHLC?pair=EURUSD&since=1594825200&interval=60")
data = response.json()



z = json.dumps(data["result"])
print(z)
data_file = open('data_file.json', 'w') 
data_file.write(z)
data_file.close()


x = pd.read_json('data_file.json')



data_file = open('data_file.csv', 'w') 
data_file.write(x['ZEURZUSD'].to_csv(header =False, index=False))
data_file.close()

fd = pd.read_csv('data_file.csv')
fd.columns = ['this']

fd['this'] = fd['this'].map(lambda x: x.strip(" '[] ").strip().rstrip(' ]'))
s = pd.concat([fd['this'].str.split(', ', expand=True)], axis=1)


data_file = open('data_file.csv', 'w') 
data_file.write(s.to_csv(header =False, index=False))
data_file.close()

final_data = pd.read_csv('data_file.csv')
print(final_data)
final_data.columns = ['1','2','3','4','5','6','7','8']
 
arr = final_data.to_numpy()
print(arr)
for x in range(np.shape(arr)[0]):
    arr[x,0] = datetime.utcfromtimestamp(arr[x,0]).strftime('%d.%m.%Y %H:%M:%S')
    print(arr)
newArr = np.delete(arr, [7,5],1)
for x in range(np.shape(arr)[0]):
    newArr[x,1] = newArr[x,1].strip("'")
for x in range(np.shape(arr)[0]):
    newArr[x,2] = newArr[x,2].strip("'")
for x in range(np.shape(arr)[0]):
    newArr[x,3] = newArr[x,3].strip("'")
for x in range(np.shape(arr)[0]):
    newArr[x,4] = newArr[x,4].strip("'")
for x in range(np.shape(arr)[0]):
    newArr[x,5] = newArr[x,5].strip("'")
    print(newArr)
dataSave = pd.DataFrame(newArr)

    

    
data_file = open('inpredo-master/src/latest.csv', 'w') 
data_file.write(dataSave.to_csv(header =False, index=False, line_terminator='\n'))
data_file.close()































