#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  31 00:21:54 2019

@author: lihepeng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Price
filename = '/home/lihepeng/Documents/Github/tmp/data/price/RtpData2017.csv'
df_price_train = pd.read_csv(filename, names=['Day','Hour','Data'])
df_price_train = df_price_train[df_price_train['Day'].between('01/01/2017', '12/31/2017')]

filename = '/home/lihepeng/Documents/Github/tmp/data/price/RtpData2018.csv'
df_price_test = pd.read_csv(filename, names=['Day','Hour','Data'])
df_price_test = df_price_test[df_price_test['Day'].between('01/01/2018', '12/31/2018')]
train_days = df_price_train.shape[0]//24
test_days = df_price_test.shape[0]//24
price_train24 = df_price_train['Data'].values.reshape(train_days,24)
price_test24 = df_price_test['Data'].values.reshape(test_days,24)

xp = np.arange(24)
x = np.linspace(0, 24, 144)
IndexTime = pd.date_range('2017-01-01 00:00:00', '2017-12-31 23:50:00', freq='10min')

price_train = np.ndarray([train_days, 144])
plt.figure()
for d in range(train_days):
    for i, price in enumerate(price_train24[d]):
        price_train[d,i*6:(i+1)*6] = price
    plt.plot(price_train[d])
plt.show()
df_price_train = pd.Series(price_train.ravel(),index=IndexTime)
price_train = df_price_train.loc['20170630 0800':'20170831 0800'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/data/price_train.txt', price_train)

price_test = np.ndarray([test_days, 144])
plt.figure()
for d in range(test_days):
    for i, price in enumerate(price_test24[d]):
        price_test[d,i*6:(i+1)*6] = price
    plt.plot(price_test[d])
plt.show()
df_price_test = pd.Series(price_test.ravel(),index=IndexTime)
price_test = df_price_test.loc['20170630 0800':'20170831 0800'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/data/price_test.txt', price_test)