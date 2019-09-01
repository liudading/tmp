#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  31 00:21:54 2019

@author: lihepeng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Temp
filename = '/home/lihepeng/Documents/Github/tmp/data/historical-hourly-weather-data/temperature.csv'
city = "Las Vegas"
df = pd.read_csv(filename)[["datetime", city]].fillna(293.4)

Kelvins_to_Fahrenheit = lambda x: (x - 273.15) * 9 / 5 + 32
df[city] = df[city].apply(Kelvins_to_Fahrenheit)

xp = np.arange(24)
x = np.linspace(0, 24, 144)

# Train
df_train = df[df["datetime"]>='2016-06-30 00:00:00'][df["datetime"]<='2016-08-31 23:00:00']
train_days = df_train.shape[0] // 24
temp24 = df_train[city].values.reshape([train_days, 24])

temp_train = np.ndarray([train_days, 144])
for d in range(train_days):
    for i, temp in enumerate(temp24[d]):
        temp_train[d,i*6:(i+1)*6] = temp

temp_train[19] = np.mean(temp_train[:15], 0)
temp_train[20] = np.mean(temp_train[16:31], 0)

plt.figure()
for d in range(train_days):
    plt.plot(temp_train[d])
plt.show()

IndexTime = pd.date_range('2017-06-30 00:00:00', '2017-08-31 23:50:00', freq='10min')
df_temp_train = pd.Series(temp_train.ravel(),index=IndexTime)
temp_train = df_temp_train.loc['20170630 2000':'20170831 2000'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/data/temp_train.txt', temp_train)

# Test
df_test = df[df["datetime"]>='2017-06-30 00:00:00'][df["datetime"]<='2017-08-31 23:00:00']
test_days = df_test.shape[0] // 24
temp24 = df_test[city].values.reshape([test_days, 24])

temp_test = np.ndarray([test_days, 144])
for d in range(test_days):
    for i, temp in enumerate(temp24[d]):
        temp_test[d,i*6:(i+1)*6] = temp

temp_test[47][48:66] = np.around(np.mean(temp_test[32:61], 0)[48:66], decimals=3)

plt.figure()
for d in range(test_days):
    plt.plot(temp_test[d])

plt.show()

IndexTime = pd.date_range('2017-06-30 00:00:00', '2017-08-31 23:50:00', freq='10min')
df_temp_test = pd.Series(temp_test.ravel(),index=IndexTime)
temp_test = df_temp_test.loc['20170630 2000':'20170831 2000'].values
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/data/temp_test.txt', temp_test)
