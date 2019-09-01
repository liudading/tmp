#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  31 00:21:54 2019

@author: lihepeng
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

xp = np.arange(24)
x = np.linspace(0, 24, 144)

# Train Price
filename = '/home/lihepeng/Documents/Github/tmp/data/price/RtpData2015.csv'
df_train = pd.read_csv(filename)
df_train = df_train[df_train['DATE'].between('06/30/2015', '08/31/2015')]
train_days = df_train.shape[0] // 24
train24 = df_train['PRICE'].values.reshape(train_days,24)

price_train = np.ndarray([train_days, 144])
plt.figure()
for d in range(train_days):
    for i, price in enumerate(train24[d]):
        price_train[d,i*6:(i+1)*6] = price
    plt.plot(price_train[d])
plt.show()

IndexTime = pd.date_range('2017-06-30 00:00:00', '2017-08-31 23:50:00', freq='10min')
df_price_train = pd.Series(price_train.ravel(),index=IndexTime)
price_train = df_price_train.loc['20170630 0800':'20170831 0800'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/data/price_train.txt', price_train)

# Test Price
filename = '/home/lihepeng/Documents/Github/tmp/data/price/RtpData2017.csv'
df_test = pd.read_csv(filename)
df_test = df_test[df_test['DATE'].between('06/30/2017', '08/31/2017')]
test_days = df_test.shape[0] // 24
test24 = df_test['PRICE'].values.reshape(test_days,24)

price_test = np.ndarray([test_days, 144])
plt.figure()
for d in range(test_days):
    for i, price in enumerate(test24[d]):
        price_test[d,i*6:(i+1)*6] = price
    plt.plot(price_test[d])
plt.show()

IndexTime = pd.date_range('2017-06-30 00:00:00', '2017-08-31 23:50:00', freq='10min')
df_price_test = pd.Series(price_test.ravel(),index=IndexTime)
price_test = df_price_test.loc['20170630 0800':'20170831 0800'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/data/price_test.txt', price_test)