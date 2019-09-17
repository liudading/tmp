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

# DDPG
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_0.1.npy'
ddpg_r_01 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_0.1.npy'
ddpg_s_01 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_1.0.npy'
ddpg_r_1 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_1.0.npy'
ddpg_s_1 = np.load(logger_dir)


logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_0.2.npy'
ddpg_r_02 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_0.2.npy'
ddpg_s_02 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_0.4.npy'
ddpg_r_04 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_0.4.npy'
ddpg_s_04 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_0.6.npy'
ddpg_r_06 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_0.6.npy'
ddpg_s_06 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_0.8.npy'
ddpg_r_08 = np.load(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_0.8.npy'
ddpg_s_08 = np.load(logger_dir)

xmax = 3000000

rolling_window = 365*1
rolling_reward = pd.Series(df_train["r"]).rolling(rolling_window)
rolling_reward = rolling_reward.mean().values[rolling_window-1:]
rolling_safety = pd.Series(df_train["s"]).rolling(rolling_window)
rolling_safety = rolling_safety.mean().values[rolling_window-1:]

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
ddpg02_r = "{0:.2f}%".format(np.mean((d+ddpg_r_02.sum())/d) * 100)
ddpg04_r = "{0:.2f}%".format(np.mean((d+ddpg_r_04.sum())/d) * 100)
ddpg06_r = "{0:.2f}%".format(np.mean((d+ddpg_r_06.sum())/d) * 100)
ddpg08_r = "{0:.2f}%".format(np.mean((d+ddpg_r_08.sum())/d) * 100)
ddpg1_r = "{0:.2f}%".format(np.mean((d+ddpg_r_1.sum())/d) * 100)
# cpo_r = "${0:.2f}".format(-df_test["r"].sum())
# ddpg01_r = "${0:.2f}".format(-ddpg_r_01.sum())
# ddpg02_r = "${0:.2f}".format(-ddpg_r_02.sum())
# ddpg04_r = "${0:.2f}".format(-ddpg_r_04.sum())
# ddpg06_r = "${0:.2f}".format(-ddpg_r_06.sum())
# ddpg08_r = "${0:.2f}".format(-ddpg_r_08.sum())
# ddpg1_r = "${0:.2f}".format(-ddpg_r_1.sum())

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)
plt.plot(np.cumsum(-df_test["r"]), label='CPO', linewidth=3.0)
plt.plot(np.cumsum(-ddpg_r_01), label=r'DDPG, $\varrho=0.1$', linewidth=3.0, linestyle='dashdot', color='#d62728')
plt.plot(np.cumsum(-ddpg_r_02), label=r'DDPG, $\varrho=0.2$', linewidth=3.0, linestyle='dotted', color='#e377c2')
plt.plot(np.cumsum(-ddpg_r_04), label=r'DDPG, $\varrho=0.4$', linewidth=3.0, color='#17becf', linestyle=linestyle_tuple['densely dashdotted'], marker='d', markersize=8, markevery=20)
plt.plot(np.cumsum(-ddpg_r_06), label=r'DDPG, $\varrho=0.6$', linewidth=3.0, color='#ff7f0e', marker='*', linestyle=linestyle_tuple['densely dashdotdotted'], markersize=10, markevery=20)
plt.plot(np.cumsum(-ddpg_r_08), label=r'DDPG, $\varrho=0.8$', linewidth=3.0, color='#2ca02c', marker='v', linestyle=linestyle_tuple['densely dashed'], markersize=7, markevery=20)
plt.plot(np.cumsum(-ddpg_r_1), label=r'DDPG, $\varrho=1.0$', linewidth=3.0, linestyle='dashed', marker='X', markersize=8, markevery=20, color='#9467bd')
plt.xlim(0, 365)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Day', fontsize=20)
plt.ylabel('Cumulative Costs ($)', fontsize=20)
plt.legend(fontsize=20)
ax.text(368, np.around(np.sum(-df_test["r"])+1,2), cpo_r, style='italic', fontsize='x-large',
        bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-ddpg_r_01),2), ddpg01_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#d62728', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-ddpg_r_02)-2,2), ddpg02_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#e377c2', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-ddpg_r_04)-0.5,2), ddpg04_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#17becf', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-ddpg_r_06)-0.5,2), ddpg06_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#ff7f0e', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-ddpg_r_08)-3,2), ddpg08_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#2ca02c', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-ddpg_r_1),2), ddpg1_r, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#9467bd', 'alpha': 0.5, 'pad': 5})
# ax.text(368, 111, dqn1_r, style='italic', fontsize='x-large',
#         bbox={'facecolor': '#17becf', 'alpha': 0.5, 'pad': 5})
ax.axis([0, 365, 0, 70])
ax.yaxis.set_label_coords(-0.11,0.5)
plt.tight_layout(rect=(0,0,1,1))
plt.show(block=False)

d = 0.1
cpo_v = "{0:.2f}%".format(np.mean(np.maximum(0, df_test["s"].values-d)/d) * 100)
ddpg01_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_01-d)/d) * 100)
ddpg02_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_02-d)/d) * 100)
ddpg04_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_04-d)/d) * 100)
ddpg06_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_06-d)/d) * 100)
ddpg08_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_08-d)/d) * 100)
ddpg1_v = "{0:.2f}%".format(np.mean(np.maximum(0, ddpg_s_1-d)/d) * 100)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)
plt.plot(np.cumsum(df_test["s"]), label='CPO', linewidth=3.0)
plt.plot(np.cumsum(ddpg_s_01), label=r'DDPG, $\varrho=0.1$', linewidth=3.0, linestyle='dashdot', color='#d62728')
plt.plot(np.cumsum(ddpg_s_02), label=r'DDPG, $\varrho=0.2$', linewidth=3.0, linestyle='dotted', color='#e377c2')
plt.plot(np.cumsum(ddpg_s_04), label=r'DDPG, $\varrho=0.4$', linewidth=3.0, color='#17becf', linestyle=linestyle_tuple['densely dashdotted'], marker='d', markersize=8, markevery=20)
plt.plot(np.cumsum(ddpg_s_06), label=r'DDPG, $\varrho=0.6$', linewidth=3.0, color='#ff7f0e', marker='*', linestyle=linestyle_tuple['densely dashdotdotted'], markersize=10, markevery=20)
plt.plot(np.cumsum(ddpg_s_08), label=r'DDPG, $\varrho=0.8$', linewidth=3.0, color='#2ca02c', marker='v', linestyle=linestyle_tuple['densely dashed'], markersize=7, markevery=20)
plt.plot(np.cumsum(ddpg_s_1), label=r'DDPG, $\varrho=1.0$', linewidth=3.0, color='#9467bd', linestyle='dashed', marker='X', markersize=8, markevery=20)
plt.xlim(0, 365)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Day', fontsize=20)
plt.ylabel('Cumulative Constraint Values (kWh)', fontsize=20)
ax.text(370, 100, cpo_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 7})
ax.text(370, np.sum(ddpg_s_01), ddpg01_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#d62728', 'alpha': 0.5, 'pad': 7})
ax.text(370, np.sum(ddpg_s_02)+100, ddpg02_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#e377c2', 'alpha': 0.5, 'pad': 7})
ax.text(370, np.sum(ddpg_s_04), ddpg04_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#17becf', 'alpha': 0.5, 'pad': 7})
ax.text(370, np.sum(ddpg_s_06), ddpg06_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#ff7f0e', 'alpha': 0.5, 'pad': 7})
ax.text(370, np.sum(ddpg_s_08), ddpg08_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#2ca02c', 'alpha': 0.5, 'pad': 7})
ax.text(370, np.sum(ddpg_s_1), ddpg1_v, style='italic', fontsize='x-large',
        bbox={'facecolor': '#9467bd', 'alpha': 0.5, 'pad': 7})
# ax.text(370, 590, dqn01_v, style='italic', fontsize='x-large',
#         bbox={'facecolor': '#8c564b', 'alpha': 0.5, 'pad': 7})
# ax.text(370, 350, dqn1_v, style='italic', fontsize='x-large',
#         bbox={'facecolor': '#17becf', 'alpha': 0.5, 'pad': 7})
ax.axis([0, 365, 0, 3600])
plt.legend(fontsize=20)
plt.tight_layout(rect=(0,0,1,1))
plt.show(block=True)