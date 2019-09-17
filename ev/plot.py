import tensorflow as tf, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from baselines.bench.monitor import load_results

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo/train'
df_train = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo/test'
df_test = load_results(logger_dir)

# TO
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/to/optimal_returns.txt'
f_to = np.loadtxt(logger_dir)

# SP
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/sp/sp_returns.txt'
f_sp = np.loadtxt(logger_dir)

# MPC
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/mpc/mpc_returns.txt'
f_mpc = np.loadtxt(logger_dir)
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/mpc/mpc_safeties.txt'
s_mpc = np.loadtxt(logger_dir)

# DDPG
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_0.1.npy'
ddpg_r_01 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_0.1.npy'
ddpg_s_01 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_1.0.npy'
ddpg_r_1 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_1.0.npy'
ddpg_s_1 = np.load(logger_dir)

# DQN
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/dqn/test/returns_0.1.npy'
dqn_r_01 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/dqn/test/safeties_0.1.npy'
dqn_s_01 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/dqn/test/returns_1.0.npy'
dqn_r_1 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/dqn/test/safeties_1.0.npy'
dqn_s_1 = np.load(logger_dir)

xmax = 3000000

rolling_window = 365*1
rolling_reward = pd.Series(df_train["r"]).rolling(rolling_window)
rolling_reward = rolling_reward.mean().values[rolling_window-1:]
rolling_safety = pd.Series(df_train["s"]).rolling(rolling_window)
rolling_safety = rolling_safety.mean().values[rolling_window-1:]

fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
plt.plot(df_train["r"], label='CPO', color='#1f77b4', alpha=0.3)
plt.plot(rolling_reward, color='#1f77b4', alpha=1.0)
plt.xlim(0, xmax)
plt.xticks(np.arange(0,xmax+1,2e5), labels=[str(i) for i in range(0,100,2)], fontsize='x-large')
plt.yticks(np.linspace(-0.8,0.8,9), fontsize='x-large')
plt.xlabel('Episode (x0.1 Million)', fontsize='x-large')
plt.ylabel('Returns ($)', fontsize='x-large')
ax1 = ax.twinx()
ax1.set_yticks(np.linspace(-0.8,0.8,9))
ax1.set_yticklabels(labels=['%.1f'%i for i in np.linspace(-0.8,0.8,9)], fontdict={'fontsize':'x-large'})
plt.tight_layout(rect=(0,0,1,1))
plt.show()


fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
axins = zoomed_inset_axes(ax, 8, loc=10)
ax.plot(df_train["s"].values, color='#ff7f0e', alpha=0.3)
ax.plot(rolling_safety, color='#ff7f0e', alpha=1.0)
ax.set_xlim(0, xmax)
ax.set_xticks(np.arange(0,xmax+1,2e5))
ax.set_xticklabels(labels=[str(i) for i in range(0,100,2)], fontdict={'fontsize':'x-large'})
ax.set_yticks(range(0,30,5))
ax.set_yticklabels(labels=[str(i) for i in range(0,30,5)], fontdict={'fontsize':'x-large'})
ax.set_xlabel('Episode (x0.1 Million)', fontsize='x-large')
ax.set_ylabel('Constraint Values (kWh)', fontsize='x-large')
# ax.axhline(0.1, linewidth=2, color='green', label='Constraint tolerance d=0.1')
# ax.legend(fontsize='x-large')
axins.plot(df_train["s"].values, color='#ff7f0e', alpha=0.3)
axins.plot(rolling_safety, color='#ff7f0e', alpha=1.0)
# axins.axhline(0.1, linewidth=2, color='green', label='Tolerance d=0.1')
x1, x2, y1, y2 = 0.0e6, 0.2e6, 0., 2 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
axins.set_xticks(np.arange(x1,x2+1,0.5e5))
axins.set_xticklabels(labels=[str(i) for i in [0,0.5,1.0,1.5,2.0]], fontdict={'fontsize':'large'})
axins.set_yticks(np.linspace(0,2.0,6))
axins.set_yticklabels(labels=['%.1f'%i for i in np.linspace(0,2.0,6)], fontdict={'fontsize':'large'})
# axins.legend(fontsize='large')
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.2")
plt.tight_layout(rect=(0.03,0.05,0.95,1))
ax.yaxis.set_label_coords(-0.12,0.5)
plt.show(block=True)

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = dict([
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


d = np.sum(f_sp)
cpo_r = "{0:.2f}%".format(np.mean((d+df_test["r"].sum())/d) * 100)
ddpg01_r = "{0:.2f}%".format(np.mean((d+ddpg_r_01.sum())/d) * 100)
ddpg1_r = "{0:.2f}%".format(np.mean((d+ddpg_r_1.sum())/d) * 100)
dqn01_r = "{0:.2f}%".format(np.mean((d+dqn_r_01.sum())/d) * 100)
dqn1_r = "{0:.2f}%".format(np.mean((d+dqn_r_1.sum())/d) * 100)
to_r = "{0:.2f}%".format(np.mean((d-f_to.sum())/d) * 100)
mpc_r = "{0:.2f}%".format(np.mean((d-f_mpc.sum())/d) * 100)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)
plt.plot(np.cumsum(-df_test["r"]), label='CPO', linewidth=3.0, linestyle=linestyle_tuple['densely dashed'])
plt.plot(np.cumsum(f_sp), label='SP', linewidth=3.0, marker='*', markersize=10, markevery=20)
plt.plot(np.cumsum(f_to), label='TO', linewidth=3.0, marker='v', markersize=7, markevery=20)
plt.plot(np.cumsum(f_mpc), label='MPC', color='k', linewidth=3.0, marker='p', markersize=8, markevery=20)
plt.plot(np.cumsum(-ddpg_r_01), label=r'DDPG, $\varrho=0.1$', linewidth=3.0, linestyle='dashdot')
plt.plot(np.cumsum(-ddpg_r_1), label=r'DDPG, $\varrho=1.0$', linewidth=3.0, linestyle='dashed', marker='X', markersize=8, markevery=20)
plt.plot(np.cumsum(-dqn_r_01), label=r'DQN, $\varrho=0.1$', linewidth=3.0, linestyle='dotted')
plt.plot(np.cumsum(-dqn_r_1), label=r'DQN, $\varrho=1.0$', linewidth=3.0, color='#17becf', linestyle=linestyle_tuple['densely dashdotted'], marker='d', markersize=8, markevery=20)
plt.xlim(0, 365)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Day', fontsize=20)
plt.ylabel('Cumulative Costs ($)', fontsize=20)
ax.text(368, 53, cpo_r, style='italic', fontsize='x-large',
        bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 5})
ax.text(368, 38, to_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#2ca02c', 'alpha': 0.5, 'pad': 5})
ax.text(368, 29, mpc_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': 'k', 'alpha': 0.5, 'pad': 5})
ax.text(368, 20, ddpg01_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#d62728', 'alpha': 0.5, 'pad': 5})
ax.text(368, 71, ddpg1_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#9467bd', 'alpha': 0.5, 'pad': 5})
ax.text(368, 62, dqn01_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#8c564b', 'alpha': 0.5, 'pad': 5})
ax.text(368, 111, dqn1_r, style='italic', fontsize='x-large',
        bbox={'facecolor': '#17becf', 'alpha': 0.5, 'pad': 5})
ax.axis([0, 365, 0, 163])
plt.legend(fontsize=20)
ax.yaxis.set_label_coords(-0.11,0.5)
plt.tight_layout(rect=(0,0,0.97,1))
plt.show()

d = 0.1
cpo_v = "{0:.2f}%".format(np.mean(np.maximum(0, df_test["s"].values-d)/d) * 100)
mpc_v = "{0:.2f}%".format(np.mean(np.maximum(0, s_mpc-d)/d) * 100)
ddpg01_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_01-d)/d) * 100)
ddpg1_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_1-d)/d) * 100)
dqn01_v = "{0:.2f}%".format(np.mean(np.maximum(0, dqn_s_01-d)/d) * 100)
dqn1_v = "{0:.2f}%".format(np.mean(np.maximum(0, dqn_s_1-d)/d) * 100)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)
plt.plot(np.cumsum(df_test["s"]), label='CPO', linewidth=3.0, linestyle=linestyle_tuple['densely dashed'])
plt.plot([0.0]*f_sp.size, linewidth=3.0)
plt.plot([0.0]*f_to.size, linewidth=3.0)
plt.plot(np.cumsum(s_mpc), label='MPC', linewidth=3.0, color='k', marker='p', markersize=8, markevery=20)
plt.plot(np.cumsum(ddpg_s_01), label=r'DDPG, $\varrho=0.1$', linewidth=3.0, linestyle='dashdot')
plt.plot(np.cumsum(ddpg_s_1), label=r'DDPG, $\varrho=1.0$', linewidth=3.0, linestyle='dashed', marker='X', markersize=8, markevery=20)
plt.plot(np.cumsum(dqn_s_01), label=r'DQN, $\varrho=0.1$', linewidth=3.0, linestyle='dotted')
plt.plot(np.cumsum(dqn_s_1), label=r'DQN, $\varrho=1.0$', linewidth=3.0, color='#17becf', linestyle=linestyle_tuple['densely dashdotted'], marker='d', markersize=8, markevery=20)
plt.xlim(0, 365)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Day', fontsize=20)
plt.ylabel('Cumulative Constraint Values (kWh)', fontsize=20)
ax.text(370, 100, cpo_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 7})
ax.text(370, 950, mpc_v, style='italic', fontsize='x-large',
        bbox={'facecolor': 'k', 'alpha': 0.5, 'pad': 7})
ax.text(370, 3450, ddpg01_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#d62728', 'alpha': 0.5, 'pad': 7})
ax.text(370, 1720, ddpg1_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#9467bd', 'alpha': 0.5, 'pad': 7})
ax.text(370, 590, dqn01_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#8c564b', 'alpha': 0.5, 'pad': 7})
ax.text(370, 350, dqn1_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#17becf', 'alpha': 0.5, 'pad': 7})
ax.axis([0, 365, 0, 3600])
plt.legend(fontsize=20)
plt.tight_layout(rect=(0,0.03,1,1))
plt.show(block=True)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
df = pd.read_csv('/home/lihepeng/Documents/Github/tmp/ev/cpo/test/results.csv', index_col=0)
df = df.loc[df.index>='2018-07-09 19']
df = df.loc[df.index<='2018-07-16 09']

fig = plt.figure(figsize=(18,5))
ax = plt.subplot(111)
ax.annotate('', xy=(157.8, 0.07083), xytext=(166, 0.07083), annotation_clip=False,
            arrowprops=dict(facecolor='black', arrowstyle='-', alpha=1.0))
ax.annotate('', xy=(157.8, 0.0437), xytext=(166, 0.0437), annotation_clip=False, 
            arrowprops=dict(color='#1f77b4', arrowstyle='-', alpha=1.0))
ax.annotate('', xy=(157.8, 0.0167), xytext=(166, 0.0167), annotation_clip=False,
            arrowprops=dict(facecolor='black', arrowstyle='-', alpha=1.0))
ax.plot(df["value"].values, color='#ff7f0e')
ax.set_xlim(0, df.shape[0])
# ax.set_ylim(0.015, 0.025)
ax.set_xticks(np.arange(0,df.shape[0]+1,20))
ax.set_xticklabels(labels=[str(i) for i in range(0,df.shape[0]+1,20)], fontdict={'fontsize':'xx-large'})
ax.set_yticks([0.02,0.03,0.04,0.05,0.06,0.07])
ax.set_yticklabels(labels=[str(i) for i in [0.02,0.03,0.04,0.05,0.06,0.07]], fontdict={'fontsize':'xx-large'}, color='#ff7f0e')
ax.set_xlabel('Time (Hour)', fontsize=20)
ax.set_ylabel('Electricity Price ($/kWh)', fontsize=20, color='#ff7f0e')

ax1 = ax.twinx()
ax1.bar(range(df["act"].shape[0]), df["act"].values)
ax1.set_yticks(range(-6,8,2))
ax1.set_yticklabels(labels=[str(i) for i in range(-6,8,2)], fontdict={'fontsize':'xx-large'}, color='#1f77b4')
ax1.set_ylabel('Energy (kWh)', fontsize=20, color='#1f77b4')
ax1.yaxis.set_label_coords(1.06,0.5)
ax1.axhline(0.0, linewidth=0.5, color='#1f77b4', label='Limit d=0.1')
ax1.text(162, 1.3, 'Charging', rotation='vertical', fontsize='xx-large', color='#1f77b4')
ax1.text(162, -6.0, 'Discharging', rotation='vertical', fontsize='xx-large', color='#1f77b4')
ax1.axis([0, df["act"].shape[0], -7, 7])
boxes = []
boxes.append(Rectangle((14, -7), 8, 14))
boxes.append(Rectangle((37, -7), 10, 14))
boxes.append(Rectangle((62, -7), 10, 14))
boxes.append(Rectangle((86, -7), 11, 14))
boxes.append(Rectangle((109, -7), 11, 14))
boxes.append(Rectangle((135, -7), 10, 14))
pc = PatchCollection(boxes, facecolor='green', alpha=0.3)
ax1.add_collection(pc)
ax1.annotate('Arrival', xy=(22, 7), xytext=(22, 8.5), annotation_clip=False,
            arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
ax1.annotate('Departure', xy=(37, 7), xytext=(37, 8.5), annotation_clip=False,
            arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
# ax1.annotate('', xy=(37, 7), xytext=(37, 8.5), annotation_clip=False,
#             arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
plt.tight_layout(rect=(0,0,1,1))
plt.show()

fig = plt.figure(figsize=(18,5))
ax = plt.subplot(111)
ax.bar(np.arange(df["act"].shape[0])+0.5,df["soc"].values*24, color='#ff7f0e')
ax.set_xticks(np.arange(0,df.shape[0]+1,20))
ax.set_xticklabels(labels=[str(i) for i in range(0,df.shape[0]+1,20)], fontdict={'fontsize':'xx-large'})
ax.set_xlim(0, df.shape[0])
ax.set_xlabel('Time (Hour)', fontsize=20)
ax.set_yticks(range(0,26,4))
ax.set_yticklabels(labels=[str(i) for i in range(0,26,4)], fontdict={'fontsize':'xx-large'})
ax.set_ylabel('Battery Energy (kWh)', fontsize=20)
ax.yaxis.set_label_coords(-0.0405,0.5)
ax.set_ylim(0, 25)
ax.axhline(24.0, linewidth=1.5, color='#1f77b4')
ax.annotate('Charging Target', xy=(32, 24), xytext=(32, 27), annotation_clip=False,
            arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
boxes = []
boxes.append(Rectangle((14, 0), 8, 25))
boxes.append(Rectangle((37, 0), 10, 25))
boxes.append(Rectangle((62, 0), 10, 25))
boxes.append(Rectangle((86, 0), 11, 25))
boxes.append(Rectangle((109, 0), 11, 25))
boxes.append(Rectangle((135, 0), 10, 25))
pc = PatchCollection(boxes, facecolor='green', alpha=0.3)
ax.add_collection(pc)
plt.tight_layout(rect=(0.0069,0,0.94,1))
plt.show()


df = pd.read_csv('/home/lihepeng/Documents/Github/tmp/ev/cpo/test/results.csv', index_col=0)
df = df.loc[df.index>='2018-12-02 19']
df = df.loc[df.index<='2018-12-09 09']

fig = plt.figure(figsize=(18,5))
ax = plt.subplot(111)
ax.annotate('', xy=(157.8, 0.070), xytext=(166, 0.070), annotation_clip=False,
            arrowprops=dict(facecolor='black', arrowstyle='-', alpha=1.0))
ax.annotate('', xy=(157.8, 0.045), xytext=(166, 0.045), annotation_clip=False, 
            arrowprops=dict(color='#1f77b4', arrowstyle='-', alpha=1.0))
ax.annotate('', xy=(157.8, 0.02), xytext=(166, 0.02), annotation_clip=False,
            arrowprops=dict(facecolor='black', arrowstyle='-', alpha=1.0))
ax.plot(df["value"].values, color='#ff7f0e')
ax.set_xlim(0, df.shape[0])
# ax.set_ylim(0.015, 0.025)
ax.set_xticks(np.arange(0,df.shape[0]+1,20))
ax.set_xticklabels(labels=[str(i) for i in range(0,df.shape[0]+1,20)], fontdict={'fontsize':'xx-large'})
ax.set_yticks([0.02,0.03,0.04,0.05,0.06,0.07])
ax.set_yticklabels(labels=[str(i) for i in [0.02,0.03,0.04,0.05,0.06,0.07]], fontdict={'fontsize':'xx-large'}, color='#ff7f0e')
ax.set_xlabel('Time (Hour)', fontsize=20)
ax.set_ylabel('Electricity Price ($/kWh)', fontsize=20, color='#ff7f0e')

ax1 = ax.twinx()
ax1.bar(range(df["act"].shape[0]), df["act"].values)
ax1.set_yticks(range(-6,8,2))
ax1.set_yticklabels(labels=[str(i) for i in range(-6,8,2)], fontdict={'fontsize':'xx-large'}, color='#1f77b4')
ax1.set_ylabel('Energy (kWh)', fontsize=20, color='#1f77b4')
ax1.yaxis.set_label_coords(1.06,0.5)
ax1.axhline(0.0, linewidth=0.5, color='#1f77b4', label='Limit d=0.1')
ax1.text(162, 1.3, 'Charging', rotation='vertical', fontsize='xx-large', color='#1f77b4')
ax1.text(162, -6.0, 'Discharging', rotation='vertical', fontsize='xx-large', color='#1f77b4')
ax1.axis([0, df["act"].shape[0], -7, 7])
boxes = []
boxes.append(Rectangle((13, -7), 10, 14))
boxes.append(Rectangle((39, -7), 9, 14))
boxes.append(Rectangle((62, -7), 9, 14))
boxes.append(Rectangle((84, -7), 12, 14))
boxes.append(Rectangle((111, -7), 8, 14))
boxes.append(Rectangle((134, -7), 9, 14))
pc = PatchCollection(boxes, facecolor='green', alpha=0.3)
ax1.add_collection(pc)
ax1.annotate('Arrival', xy=(23, 7), xytext=(23, 8.5), annotation_clip=False,
            arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
ax1.annotate('Departure', xy=(39, 7), xytext=(39, 8.5), annotation_clip=False,
            arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
# ax1.annotate('', xy=(37, 7), xytext=(37, 8.5), annotation_clip=False,
#             arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
plt.tight_layout(rect=(0,0,1,1))
plt.show()

fig = plt.figure(figsize=(18,5))
ax = plt.subplot(111)
ax.bar(np.arange(df["act"].shape[0])+0.5,df["soc"].values*24, color='#ff7f0e')
ax.set_xticks(np.arange(0,df.shape[0]+1,20))
ax.set_xticklabels(labels=[str(i) for i in range(0,df.shape[0]+1,20)], fontdict={'fontsize':'xx-large'})
ax.set_xlim(0, df.shape[0])
ax.set_xlabel('Time (Hour)', fontsize=20)
ax.set_yticks(range(0,26,4))
ax.set_yticklabels(labels=[str(i) for i in range(0,26,4)], fontdict={'fontsize':'xx-large'})
ax.set_ylabel('Battery Energy (kWh)', fontsize=20)
ax.yaxis.set_label_coords(-0.0405,0.5)
ax.set_ylim(0, 25)
ax.axhline(24.0, linewidth=1.5, color='#1f77b4')
ax.annotate('Charging Target', xy=(32, 24), xytext=(32, 27), annotation_clip=False,
            arrowprops=dict(facecolor='black', alpha=1.0), fontsize='xx-large')
boxes = []
boxes.append(Rectangle((13, 0), 10, 25))
boxes.append(Rectangle((39, 0), 9, 25))
boxes.append(Rectangle((62, 0), 9, 25))
boxes.append(Rectangle((84, 0), 12, 25))
boxes.append(Rectangle((111, 0), 8, 25))
boxes.append(Rectangle((134, 0), 9, 25))
pc = PatchCollection(boxes, facecolor='green', alpha=0.3)
ax.add_collection(pc)
plt.tight_layout(rect=(0.0069,0,0.94,1))
plt.show()