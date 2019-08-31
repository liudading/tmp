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

pricefile = '~/Documents/Github/data/RtpData2017.csv'
df_train = pd.read_csv(pricefile, names=['date','hour','value'])
df_train['value'] = df_train['value'].astype('float32')
df_train.index = pd.date_range('2017-01-01-00', '2017-12-31-23', freq='1H')

pricefile = '~/Documents/Github/data/RtpData2018.csv'
df_test = pd.read_csv(pricefile, names=['date','hour','value'])
df_test['value'] = df_test['value'].astype('float32')
df_test.index = pd.date_range('2017-12-31-00', '2019-01-01-23', freq='1H')

def optimize(ep_prices, init_soc):
    model = Model("ev_charging")

    # Variables
    B, CH_P, DI_P, P, SOC, OBJ = {}, {}, {}, {}, {}, {}
    T = ep_prices.size
    for t in range(T):
        B[t] = model.addVar(vtype='B', name='B_%s' % t)
        CH_P[t] = model.addVar(vtype='C', name='CH_P_%s' % t, lb=0, ub=MAX_CHARGING_POWER)
        DI_P[t] = model.addVar(vtype='C', name='DI_P_%s' % t, lb=MAX_DISCHARGING_POWER, ub=0)
        P[t] = model.addVar(vtype='C', name='P_%s' % t, lb=MAX_DISCHARGING_POWER, ub=MAX_CHARGING_POWER)
        SOC[t] = model.addVar(vtype='C', name='SOC_%s' % t, lb=MIN_SOC, ub=MAX_SOC)
        OBJ[t] = model.addVar(vtype='C', name='OBJ_%s' % t, lb=-1e10, ub=1e10)

    # Constraints
    for t in range(T):
        model.addCons(P[t] == B[t]*CH_P[t]*CHARGING_EFFICIENCY + (1-B[t])*DI_P[t]/DISCHARGING_EFFICIENCY)
        if t == 0:
            model.addCons(SOC[t] == init_soc + P[t] / CAPACITY, name='SOC_DYN_0')
        else:
            model.addCons(SOC[t] == SOC[t-1] + P[t] / CAPACITY, name='SOC_DYN_(%s)' % t)
        model.addCons(OBJ[t] == ep_prices[t] * P[t], name='Objective(%s)' % t)
    model.addCons(SOC[T-1] == TARGET_SOC, name='Target')

    # Set objective function
    model.setObjective(quicksum([OBJ[t] for t in range(T)]), 'minimize')

    # Execute optimization
    model.hideOutput()
    # model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
    model.optimize()

    # Get results
    p = [model.getVal(P[t]) for t in range(T)]
    soc = [model.getVal(SOC[t]) for t in range(T)]
    f = model.getObjVal()

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
for arr_date in price['date'].unique()[1:-1]:
    arr_hour = str(int(np.round(np.clip(rnd.normal(18,1),15,21)))).zfill(2)
    dep_hour = str(int(np.round(np.clip(rnd.normal(8,1),6,11)))).zfill(2)
    arr_time = pd.to_datetime(arr_date+' '+arr_hour)
    dep_time = pd.to_datetime(arr_date+' '+dep_hour) + pd.Timedelta(days=1)

    ep_prices = price.loc[arr_time:dep_time-pd.Timedelta(hours=1)]["value"].values
    init_soc = np.clip(rnd.normal(0.5, 0.1), 0.2, 0.8)
    f, p, soc = optimize(ep_prices, init_soc)
    f_vals.append(f)
#     plot(ep_prices, soc)

print(np.mean(f_vals))
plt.plot(f_vals)
plt.show()

np.savetxt('/home/lihepeng/Documents/Github/tmp/ev/to/optimal_returns.txt', f_vals)