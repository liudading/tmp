import tensorflow as tf, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from baselines.bench.monitor import load_results

logger_dir = '/home/lihepeng/Documents/Github/tmp/dr_isgt/ppo/train'
df_train = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/dr_isgt/ppo/test'
df_test = load_results(logger_dir)

xmax = 50000

rolling_window = 20*1
rolling_reward = pd.Series(df_train["r"]).rolling(rolling_window)
rolling_reward = rolling_reward.mean().values[rolling_window-1:]

fig = plt.figure(figsize=(8,4))
ax = plt.subplot(111)
# plt.plot(df_train["r"], label='PPO', color='#1f77b4', alpha=0.3)
plt.plot(rolling_reward, color='#1f77b4', alpha=1.0)
plt.xlim(0, xmax)
plt.ylim(-3.5,0.0)
plt.xticks(np.arange(0,xmax+1,5000), labels=[str(i) for i in range(0,51,5)], fontsize='x-large')
plt.yticks(np.linspace(-3.5,0.0,8), labels=['%.1f'%i for i in np.linspace(-3.5,0.0,8)], fontsize='x-large')
plt.xlabel('Episode (x 1000)', fontsize='xx-large')
plt.ylabel('Cumulative Rewards', fontsize='xx-large')
ax1 = ax.twinx()
ax1.set_ylim(-3.5,0.0)
ax1.set_yticks(np.linspace(-3.5,0.0,8))
ax1.set_yticklabels(labels=['%.1f'%i for i in np.linspace(-3.5,0.0,8)], fontdict={'fontsize':'x-large'})
plt.tight_layout(rect=(0,0,1,1))
plt.show()

""" Scheduling Results """
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

T, day = 96, 191 # 174, 180, 182, 195, 198, 208, 235, 236

df = pd.read_csv('/home/lihepeng/Documents/Github/tmp/dr_isgt/ppo/test/results.csv', index_col=0)
df = df.loc[df.index>=day*96]
df = df.loc[df.index<(day+1)*96]

x = range(T)
xticks = range(0, T+1, 8)
xticklabels = [str(i) for i in range(8,24,2)] + [str(i) for i in range(0,9,2)]

n_subfigs = 6
plt.close()
plt.figure(figsize=(8,10))

ax1 = plt.subplot(n_subfigs,1,1)
ax1.step(x, df["Price"], label='Electricity Price', color='#ff7f0e', linewidth=2)
ax1.set_xticks(xticks)
ax1.set_xticklabels(labels=xticklabels, fontsize='large')
ax1.set_yticks(np.linspace(0.02,0.06,3))
ax1.set_yticklabels(labels=['%.2f'%i for i in np.linspace(0.02,0.06,3)], fontsize='large')
ax1.set_ylabel('Price ($/kWh)', fontsize='x-large')
ax1.set_xlim(0,T)
ax1.legend(fontsize='x-large')

ax2 = plt.subplot(n_subfigs,1,2)
ax2.step(x, df["P_dw"], label='Dishwasher', linewidth=2)
ax2.set_xticks(xticks)
ax2.set_xticklabels(labels=xticklabels, fontsize='large')
ax2.set_yticks(np.linspace(0.0,0.5,3))
ax2.set_yticklabels(labels=['%.2f'%i for i in np.linspace(0.0,0.5,3)], fontsize='large')
ax2.axvline(df["A_dw"].iloc[0], c='r')
ax2.axvline(df["B_dw"].iloc[0], c='r')
ax2.set_xlim(0,T)
ax2.set_ylabel('Power (kW)', fontsize='x-large')
ax2.yaxis.set_label_coords(-0.1,0.5)
ax2.legend(fontsize='x-large')
boxes = []
boxes.append(Rectangle((df["A_dw"].iloc[0], -0.1), df["B_dw"].iloc[0]-df["A_dw"].iloc[0], 25))
pc = PatchCollection(boxes, facecolor='green', alpha=0.05)
ax2.add_collection(pc)

ax3 = plt.subplot(n_subfigs,1,3)
ax3.step(x, df["P_wm"], label='Washing Machine', linewidth=2)
ax3.set_xticks(xticks)
ax3.set_xticklabels(labels=xticklabels, fontsize='large')
ax3.set_yticks(np.linspace(0.0,0.4,3))
ax3.set_yticklabels(labels=['%.2f'%i for i in np.linspace(0.0,0.4,3)], fontsize='large')
ax3.axvline(df["A_wm"].iloc[0], c='r')
ax3.axvline(df["B_wm"].iloc[0], c='r')
ax3.set_xlim(0,T)
ax3.set_ylabel('Power (kW)', fontsize='x-large')
ax3.yaxis.set_label_coords(-0.1,0.5)
ax3.legend(fontsize='x-large')
boxes = []
boxes.append(Rectangle((df["A_wm"].iloc[0], -0.1), df["B_wm"].iloc[0]-df["A_wm"].iloc[0], 25))
pc = PatchCollection(boxes, facecolor='green', alpha=0.05)
ax3.add_collection(pc)

ax4 = plt.subplot(n_subfigs,1,4)
ax4.step(x, df["P_cd"], label='Clothes Dryer', linewidth=2)
ax4.set_xticks(xticks)
ax4.set_xticklabels(labels=xticklabels, fontsize='large')
ax4.set_yticks(np.linspace(0.0,1.0,3))
ax4.set_yticklabels(labels=['%.1f'%i for i in np.linspace(0.0,1.0,3)], fontsize='large')
ax4.axvline(df["A_cd"].iloc[0], c='r')
ax4.axvline(df["B_cd"].iloc[0], c='r')
ax4.set_xlim(0,T)
ax4.set_ylabel('Power (kW)', fontsize='x-large')
ax4.yaxis.set_label_coords(-0.1,0.5)
ax4.legend(fontsize='x-large')
boxes = []
boxes.append(Rectangle((df["A_cd"].iloc[0], -0.1), df["B_cd"].iloc[0]-df["A_cd"].iloc[0], 25))
pc = PatchCollection(boxes, facecolor='green', alpha=0.05)
ax4.add_collection(pc)

ax5 = plt.subplot(n_subfigs,1,5)
ax5.step(x, df["P_st"], label='Stove', linewidth=2)
ax5.set_xticks(xticks)
ax5.set_xticklabels(labels=xticklabels, fontsize='large')
ax5.axvline(df["A_st"].iloc[0], c='r')
ax5.axvline(df["B_st"].iloc[0], c='r')
ax5.set_xlim(0,T)
ax5.set_yticks(range(3))
ax5.set_yticklabels(labels=['%.1f'%i for i in range(3)], fontsize='large')
ax5.set_ylabel('Power (kW)', fontsize='x-large')
ax5.yaxis.set_label_coords(-0.1,0.5)
ax5.legend(fontsize='x-large')
boxes = []
boxes.append(Rectangle((df["A_st"].iloc[0], -0.1), df["B_st"].iloc[0]-df["A_st"].iloc[0], 25))
pc = PatchCollection(boxes, facecolor='green', alpha=0.05)
ax5.add_collection(pc)

# ax6 = plt.subplot(n_subfigs,1,6)
# ln1 = ax6.step(x, df["P_fg"], label='Frigerator')
# ln2 = ax6.step(x, df["P_tv"], label='TV')
# ln3 = ax6.step(x, df["P_lg"], label='Lights')
# ax6.set_xticks(xticks)
# ax6.set_xticklabels(labels=xticklabels, fontsize='large')
# ax6.set_xlim(0,T+1)
# ax6.set_ylim(-0.02,0.22)
# ax6.set_yticks(np.linspace(0.0,0.2,2))
# ax6.set_yticklabels(labels=['%.1f'%i for i in np.linspace(0.0,0.2,2)], fontsize='large')
# ax6.set_ylabel('Power (kW)', fontsize='x-large')
# ax6.yaxis.set_label_coords(-0.1,0.5)
# lns = ln1+ln2
# labs = [l.get_label() for l in lns]
# ax6.legend(lns, labs, ncol=1, loc=2, fontsize='x-large')
# ax61 = ax6.twinx()
# lns = ln3
# labs = [l.get_label() for l in lns]
# ax61.legend(lns, labs, ncol=1, loc=1, fontsize='x-large')

ax6 = plt.subplot(n_subfigs,1,6)
ln1 = ax6.step(x, df["P_ev"].values, label='EV power', linewidth=2)
ax6.set_xticks(xticks)
ax6.set_xticklabels(labels=xticklabels, fontsize='large')
ax6.axvline(df["A_ev"].iloc[0], c='r')
ax6.axvline(df["B_ev"].iloc[0], c='r')
ax6.set_xlim(0,T+1)
ax6.set_ylabel('Power (kW)', fontsize='x-large')
ax6.set_yticks(np.linspace(-6.0,6.0,3))
ax6.set_yticklabels(labels=['%.1f'%i for i in np.linspace(-6.0,6.0,3)], fontsize='large')
ax6.yaxis.set_label_coords(-0.1,0.5)
boxes = []
boxes.append(Rectangle((df["A_ev"].iloc[0], -6.1), df["B_ev"].iloc[0]-df["A_ev"].iloc[0], 25))
pc = PatchCollection(boxes, facecolor='green', alpha=0.05)
ax6.add_collection(pc)
ax61 = ax6.twinx()
df["SoC"].iloc[int(df["A_ev"].iloc[0])]=0
ln2 = ax61.step(x, df["SoC"]*24, color='m', label='EV Battery Energy', linewidth=2)
ax61.set_yticks(np.linspace(0.0,24.0,3))
ax61.set_yticklabels(labels=['%.0f'%i for i in np.linspace(0.0,24.0,3)], fontsize='large')
ax61.set_ylabel('Energy (kWh)', fontsize='x-large')

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax6.legend(lns, labs, ncol=1, loc=2, fontsize='large')

plt.tight_layout()
plt.show(block=True)
plt.pause(0.5)


""" Comparison """
# optimal
logger_dir = '/home/lihepeng/Documents/Github/tmp/dr_isgt/optimal/test_costs.txt'
f_opt = np.loadtxt(logger_dir)

# nodr
logger_dir = '/home/lihepeng/Documents/Github/tmp/dr_isgt/nodr/test_costs.txt'
f_nodr = np.loadtxt(logger_dir)

# PPO
logger_dir = '/home/lihepeng/Documents/Github/tmp/dr_isgt/ppo/test/test_costs.txt'
f_ppo = np.loadtxt(logger_dir)

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


d = np.sum(f_nodr)
ppo_r = "{0:.2f}%".format(np.mean((d-f_ppo.sum())/d) * 100)
opt_r = "{0:.2f}%".format(np.mean((d-f_opt.sum())/d) * 100)

fig = plt.figure(figsize=(8*1.2,4*1.2))
ax = plt.subplot(111)
plt.plot(np.cumsum(f_ppo), label='Proposed approach', linewidth=2.0, c='r')
plt.plot(np.cumsum(f_nodr), label='Without DR', linewidth=2.0, marker='*', markersize=10, markevery=20)
plt.plot(np.cumsum(f_opt), label='Optimal', linewidth=2.0, marker='v', markersize=7, markevery=20, c='g', linestyle=linestyle_tuple['densely dashed'])
plt.xlim(0, 363)
plt.ylim(0,260)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Test Day', fontsize=20)
plt.ylabel('Cumulative Costs ($)', fontsize=20)
ax.text(367, 160, ppo_r, style='italic', fontsize='x-large',
        bbox={'facecolor': 'r', 'alpha': 0.5, 'pad': 5})
ax.text(367, 135, opt_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': 'g', 'alpha': 0.5, 'pad': 5})
plt.legend(fontsize=20)
ax.yaxis.set_label_coords(-0.11,0.5)
plt.tight_layout()
plt.show()