import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum

CAPACITY = 24
MAX_CHARGING_POWER = 6
MAX_DISCHARGING_POWER = -6
CHARGING_EFFICIENCY = 0.98
DISCHARGING_EFFICIENCY = 0.98
MAX_HORIZION = 24
DELTA_T = 1
MAX_SOC = 1.0
MIN_SOC = 0.1
TARGET_SOC = 1.0

seed = 1314
rnd = np.random.RandomState(seed)

pricefile = '~/Documents/Github/tmp/data/price/RtpData2017.csv'
df_2017 = pd.read_csv(pricefile)
df_2017['PRICE'] = df_2017['PRICE'].astype('float32')
df_2017.index = pd.date_range('2017-01-01-00', '2017-12-31-23', freq='1H')

pricefile = '~/Documents/Github/tmp/data/price/RtpData2018.csv'
df_2018 = pd.read_csv(pricefile)
df_2018['PRICE'] = df_2018['PRICE'].astype('float32')
df_2018.index = pd.date_range('2018-01-01-00', '2018-12-31-23', freq='1H')

pricefile = '~/Documents/Github/tmp/data/price/RtpData2019.csv'
df_2019 = pd.read_csv(pricefile)
df_2019['PRICE'] = df_2019['PRICE'].astype('float32')
df_2019.index = pd.date_range('2019-01-01-00', '2019-08-31-23', freq='1H')

df_train = df_2017
df_test = pd.concat([df_2017[-24:], df_2018, df_2019[:24]])

def charge(ep_prices, init_soc):
    p, soc, f = [], [], 0.0
    T = ep_prices.size
    soc_t = init_soc
    for t in range(T):
        if soc_t < TARGET_SOC:
            P_t = min(MAX_CHARGING_POWER, (TARGET_SOC - soc_t) * CAPACITY)
        else:
            P_t = 0.0
        soc_t = soc_t + P_t / CAPACITY
        f += P_t * ep_prices[t]
        p.append(P_t)
        soc.append(soc_t)

    return f, p, soc

def plot(ep_prices, soc):
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(ep_prices, label='Price')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(soc, c='#ff7f0e', label='SOC')

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=1, fontsize='x-large')

    plt.tight_layout(rect=[0,0,0.99,1.0])
    plt.show(block=True)
    plt.pause(.1)

# choose price data
price = {"train": df_train, "test": df_test}["test"]
f_vals = []
for arr_date in price['DATE'].unique()[1:-1]:
    arr_hour = str(int(np.round(np.clip(rnd.normal(18,1),15,21)))).zfill(2)
    dep_hour = str(int(np.round(np.clip(rnd.normal(8,1),6,11)))).zfill(2)
    arr_time = pd.to_datetime(arr_date+' '+arr_hour)
    dep_time = pd.to_datetime(arr_date+' '+dep_hour) + pd.Timedelta(days=1)

    ep_prices = price.loc[arr_time:dep_time-pd.Timedelta(hours=1)]["PRICE"].values
    init_soc = np.clip(rnd.normal(0.5, 0.1), 0.2, 0.8)
    f, p, soc = charge(ep_prices, init_soc)
    f_vals.append(f)
    # plot(ep_prices, soc)

print(np.mean(f_vals))
plt.plot(f_vals)
plt.show()

np.savetxt('/home/lihepeng/Documents/Github/tmp/ev/sp/sp_returns.txt', f_vals)