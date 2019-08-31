import numpy as np
import pandas as pd
price_2018 = pd.read_csv('/home/lihepeng/Documents/Github/tmp/data/price/RtpData2018.csv')
price_2017 = price_2018
price_2017_Aug31_to_Dec31 = pd.read_csv('/home/lihepeng/Documents/Github/tmp/data/price/RtpData2017_Aug31_to_Dec31.csv')

price_2017_txt = np.loadtxt('/home/lihepeng/Documents/Github/tmp/dr_isgt/price_train.txt')
IndexTime = pd.date_range('2017-01-01 08:00:00', '2017-12-31 08:00:00', freq='15min')
price_train = pd.Series(price_2017_txt,index=IndexTime)

price_2018_txt = np.loadtxt('/home/lihepeng/Documents/Github/tmp/dr_isgt/price_test.txt')
IndexTime = pd.date_range('2018-01-01 08:00:00', '2018-12-31 08:00:00', freq='15min')
price_test = pd.Series(price_2018_txt,index=IndexTime)

price_2017_list = []
for i, price in enumerate(price_2017_txt):
    if (i+1) % 4 == 0:
        price_2017_list.append(price)

price_2017_array = np.array(price_2017_list)
mean_8hr = np.mean(np.reshape(price_2017_array[16:-8], [363,24])[:,0:8], axis=0)
recovered_price_2017 = np.hstack([np.around(mean_8hr, decimals=5), price_2017_array])

for index, row in price_2017.iterrows():
    print(index)
    if row["DATE"] >= '09/01/2018':
        price_2017.loc[index, "DATE"] = (row["DATE"][:-1]+'7')
        price_2017_today = price_2017_Aug31_to_Dec31[price_2017_Aug31_to_Dec31["DATE"]==(row["DATE"][:-1]+'7')]
        price_2017.loc[index, "PRICE"] = price_2017_today[price_2017_today["HOUR"]==row["HOUR"]]["PRICE"].values[0]
    else:
        price_2017.loc[index, "DATE"] = (row["DATE"][:-1]+'7')
        price_2017.loc[index, "PRICE"] = recovered_price_2017[index]

price_2017.to_csv('/home/lihepeng/Documents/Github/tmp/data/price/RtpData2017.csv')