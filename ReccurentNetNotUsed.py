#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import os
from sklearn import preprocessing
import time
from datetime import datetime
import datetime
from datetime import date
from collections import deque
from datetime import timezone
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
df = pd.read_csv("C:/Users/natha/Documents/ResearcProject/inpredo-master/financial_data/eurUsdss.csv", names=["time", "low", "high","open", "close", "volume"])


# In[19]:




#Data using trade openings instead of close 
#financialData = pd.read_csv('C:/Users/natha/Documents/ResearcProject/inpredo-master/financial_data/eurUsdss2.csv', names=["Date", "Price"])




# RSI Relative Strength Index, a technical indicator to chart strength and weakness of historical data and current data
# Formula RSI = 100 – [100 / ( 1 + (Average of Upward Price Change / Average of Downward Price Change )
# Source https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI
def Rsi(price_series, period=14, method='ema'):
    '''
    Relative strength index
    '''

    delta = price_series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    if method == 'ema':
        # Calculate the EWMA RSI
        roll_up = up.ewm(com=period, min_periods=period, adjust=True, ignore_na=False).mean()
        roll_down = down.abs().ewm(com=period, min_periods=period, adjust=True, ignore_na=False).mean()
    elif method == 'sma':
        # Calculate the SMA RSI
        roll_up = up.rolling(window=period, min_periods=period, center=False).mean()
        roll_down = down.abs().rolling(window=period, min_periods=period, center=False).mean()

    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))

    return RSI


# Add column for RSI to use as a feature for learning
df['rsi'] = Rsi(df['close'])
environment_columns = ['rsi', 'close']

# Drop NA columns and reset index
df.dropna(inplace=True)

print(df)

df['close'].plot(figsize=(16,6))
df['rsi'].plot(figsize=(16,6))


#Data Would show not normaly distributed, however the price column is not an accurate representation of distribution due to the rate of inflation and various other factors 
plt.hist(df['close'])

#using RSI We can get a more accurate represntation and as expected when using rsi the data then shows as normally distributed 
plt.hist(df['rsi'])







print(df)
SEQ_len = 60
future_period_predict = 3
RATIO_TO_PREDICT = "EUR-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_len}-SEQ-{future_period_predict}-PRED-{int(time.time())}"

main_df = pd.DataFrame()
df['low'] = df['low'].astype(float)
df['high'] = df['high'].astype(float)
df['open'] = df['open'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(float)
df['time'] = df['time'].astype(str)
print(df.dtypes)


# In[20]:



indexNames = df[ (df['volume'] < 999)].index
df.drop(indexNames , inplace=True)
df = df.reset_index(drop=True)
print(len(df))
print(df)


# In[21]:


def dateToTimestamp(neD):
    test = str(neD)
    f = "%m/%d/%Y %H:%M"
    date = datetime.datetime.strptime(test, f)
    timestamp = datetime.datetime.timestamp(date)
    return(timestamp)


# In[22]:


for i in df.index:
        x = dateToTimestamp(df.iloc[i,0])
        df.at[i, 'time'] = int(x)
        print(i)


# In[23]:


print(df.index)


# In[ ]:





# In[24]:


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0 

def preprocess_df(funf):
    fundf = funf.drop('future',1)
    
    for col in fundf.columns[1:]:
        
        if col != "target":
         
            fundf[col] = fundf[col].pct_change()
            fundf.dropna(inplace=True)
            fundf[col] = preprocessing.scale(fundf[col].values)
            print(fundf[col].values)
            
   
    fundf.dropna(inplace= True)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_len)
    print(fundf.head())
    for c in fundf.values:
        prev_days.append([n for n in c[:-1]])
        if len(prev_days) == SEQ_len:
            sequential_data.append([np.array(prev_days), c[-1]])
    random.shuffle(sequential_data)
    
    
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0: 
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    X = []
    Y = []

    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X), Y
    #for i in df.values


# In[ ]:





# In[25]:



#ratios = ["","","",""]
#for ratio in ratios:
 #   dataset = f"data/{ratio}.csv"
  #  df = pd.read_csv(dataset, names =["time", "low","high","open", "close", "volume"])
   # print(df.head())


# In[26]:


newDf = df[['time', 'close', 'volume']].copy()


# In[27]:


print(newDf)


# In[28]:


df['future'] = df["close"].shift(-future_period_predict)

print(df.head())


# In[29]:


print(df)


# In[30]:


df['target'] = list(map(classify, df["close"],df["future"] ))
print(df[["close", "future", "target"]].head())


# In[31]:



df


# In[ ]:





# In[32]:



times = sorted(df['time'].index.values)
last_5pct = times[-int(0.05*len(times))]

validation_df = df[(df.index >= last_5pct)]
df = df[(df.index < last_5pct)]

train_x, train_y = preprocess_df(df)
validation_x, validation_y = preprocess_df(validation_df)


# In[ ]:





# In[ ]:





# In[ ]:







# In[33]:


model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
          
model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
             optimizer=opt,
             metrics =['accuracy'])
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbodes=1, save_best_only=True, mode='max'))

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y))


# In[ ]:



history = model.fit(
    train_x, train_y,
    batch_size=1,
    epochs = 1,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint])


# In[ ]:





# In[ ]:





# In[ ]:




