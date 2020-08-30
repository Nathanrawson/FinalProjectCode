
"""
Created on Wed Jul  1 21:45:23 2020

@author: natha
"""
import fxcmpy
import socketio
import pandas as pd
import sched, time
from random import randrange
import datetime
from sklearn.naive_bayes import ComplementNB


df = pd.read_csv("EurUsd.csv", names=["time", "low", "high","open", "close", "volume"])
df['low'] = df['low'].astype(float)
df['high'] = df['high'].astype(float)
df['open'] = df['open'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(float)
df['time'] = df['time'].astype(str)

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

indexNames = df[ (df['volume'] < 999)].index
df.drop(indexNames , inplace=True)
df = df.reset_index(drop=True)

   #Adding future column
df['future'] = df["close"].shift(-2)
#running classifiy function to create target column
df['target'] = list(map(classify, df["close"],df["future"] ))

newXDf = df.drop(columns=['future'])
print('number of buy rows are '+ str(len(newXDf[newXDf['target'] == 0])))
print('number of sell rows are '+ str(len(newXDf[newXDf['target'] == 1])))
X_Train = newXDf

#Making Buy snd Sell the same length of 70,000 for buy and 70,000 for sell
sell = X_Train[X_Train['target'] == 1].iloc[0:80000]
buy = X_Train[X_Train['target'] == 0].iloc[0:80000]
categories = [buy,sell]
newYDf = pd.concat(categories)
tempSet = pd.concat(categories)
trainingSet = tempSet[['time', 'low', 'high', 'close','volume', 'target']].copy()


#Shuffling the data below
trainingSet = trainingSet.sample(frac=1).reset_index(drop=True)
y = trainingSet['target']
X_Train = trainingSet
X_Train = X_Train.drop(columns=['target'])

clf = ComplementNB()
X = X_Train[['close', 'low']].copy()
clf.fit(X, y)


#ik7Fv
#D25823582

print("Running")

#data = con.get_candles('EUR/USD', period='m1', number=250)

def SetStopLosses(con):
    openIds = con.get_open_positions()['tradeId']
    for x in range (len(openIds)):
        print(openIds[x])
        con.change_trade_stop_limit(openIds[x], is_stop=True, rate=-5)
        con.change_trade_stop_limit(openIds[x], is_in_pips=False,is_stop=False, rate=+20)

def Predict(con):
     orders = con.get_open_positions().T
     if(len(orders)>0):
         previousOrder = orders[0]
         closePrice = previousOrder['close']
         openPrice = previousOrder ['open']
         openPriceArr = [openPrice]
         closePriceArr = [closePrice]
         print(closePrice)
         print(openPrice)
         dataset = pd.DataFrame({'Column1': closePriceArr, 'Column2': openPriceArr})
         print(dataset)
         y_pred = clf.predict(dataset)
         targetVal = y_pred[0]
         print('predicted value is' + str(targetVal))
     else:
         targetVal = 0
     return targetVal

def CalculateInvestment(con):
    orders = con.get_open_positions().T
    prevAmount = 1
    profitLoss = 1
    print(len(orders))
    if len(orders) > 0:
        previousOrder = orders[0]
        profitLoss = previousOrder['grossPL']
        prevAmount = previousOrder['amountK']
        print(profitLoss)
        print(prevAmount)
    if profitLoss < 0:
        print('amount should be doubling')
        amount = prevAmount * 2.1
        print("I should be going in with" + str(amount))
    else:
        print(str(profitLoss)+ ' is not less than 0')
        amount = 3
    print('Actual')
    print(amount)
    return amount

def MakeOrder(action, amount, con):
    amount = int(amount)
    print(amount)
    if action == "buy":
        curAmount = amount
        print(amount)
        con.create_market_buy_order('EUR/USD', curAmount)
    elif action == "sell":
        print(amount)
        curAmount = amount
        con.create_market_sell_order('EUR/USD', curAmount)
    GetUpdate(con)
    del amount
    del curAmount
    #SetStopLosses()

def GetUpdate(con):
    print(con.get_open_positions().T)

s = sched.scheduler(time.time, time.sleep)

def MakeTrade(sc):
    TOKEN = '6817a5e9c93ad9c62201c9e032a0e4d4f58c0d59'
    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='demo', log_file='log.txt')

    print("Creating Order")
    #datetime.datetime.today().weekday()
    weekNum = datetime.datetime.today().weekday()
    print('Below is prev order')
    print(con.get_open_positions().T)
    amount = CalculateInvestment(con)
    print(amount)
    num = Predict(con)
    if num == 0 and weekNum < 5:
        print("Buying")
        print()
        con.close_all_for_symbol('EUR/USD')
        MakeOrder("buy", amount, con)
    elif num == 1 and weekNum < 5:
        print("Selling")
        con.close_all_for_symbol('EUR/USD')
        MakeOrder("sell", amount, con)
    else:
        print("No Order to be made")
    del amount
    s.enter(1800, 1, MakeTrade, (sc,))
    con.close()
s.enter(1800, 1, MakeTrade, (s,))
s.run()
print("Another trade being made")

con.close_all_for_symbol('EUR/USD')

con.close()