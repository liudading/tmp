#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:30:54 2019

@author: lihepeng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Price
filename = '/home/lihepeng/Documents/Github/tmp/data/price/RtpData2017.csv'
df_price_train = pd.read_csv(filename)
df_price_train = df_price_train[df_price_train['DATE'].between('01/01/2017', '12/31/2017')]

filename = '/home/lihepeng/Documents/Github/tmp/data/price/RtpData2018.csv'
df_price_test = pd.read_csv(filename)
df_price_test = df_price_test[df_price_test['DATE'].between('01/01/2018', '12/31/2018')]

train_days = df_price_train.shape[0]//24
test_days = df_price_test.shape[0]//24
price_train24 = df_price_train['PRICE'].values.reshape(train_days,24)
price_test24 = df_price_test['PRICE'].values.reshape(test_days,24)

xp = np.arange(24)
x = np.linspace(0, 24, 96)
IndexTime = pd.date_range('2017-01-01 00:00:00', '2017-12-31 23:45:00', freq='15min')

price_train = np.ndarray([train_days, 96])
plt.figure()
for d in range(train_days):
    for i, price in enumerate(price_train24[d]):
        price_train[d,i*4:(i+1)*4] = price
    plt.plot(price_train[d])
plt.show()
df_price_train = pd.Series(price_train.ravel(),index=IndexTime)
price_train = df_price_train.loc['20170101 0800':'20171231 0800'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr_isgt/price_train.txt', price_train)

price_test = np.ndarray([test_days, 96])
plt.figure()
for d in range(test_days):
    for i, price in enumerate(price_test24[d]):
        price_test[d,i*4:(i+1)*4] = price
    plt.plot(price_test[d])
plt.show()
df_price_test = pd.Series(price_test.ravel(),index=IndexTime)
price_test = df_price_test.loc['20170101 0800':'20171231 0800'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr_isgt/price_test.txt', price_test)