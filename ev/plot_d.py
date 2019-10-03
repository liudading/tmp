import tensorflow as tf, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from baselines.bench.monitor import load_results

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo/train'
df_train = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo/test'
df_test = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo_v1/train/cpo_d_is_1'
df_train_d_is_1 = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo_v1/test/cpo_d_is_1'
df_test_d_is_1 = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo_v1/train/cpo_d_is_2'
df_train_d_is_2 = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo_v1/test/cpo_d_is_2'
df_test_d_is_2 = load_results(logger_dir)

logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/sp/sp_returns.txt'
f_sp = np.loadtxt(logger_dir)

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
cpo_r_d_is_1 = "{0:.2f}%".format(np.mean((d+df_test_d_is_1["r"].sum())/d) * 100)
cpo_r_d_is_2 = "{0:.2f}%".format(np.mean((d+df_test_d_is_2["r"].sum())/d) * 100)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)
plt.plot(np.cumsum(f_sp), label='SP', linewidth=3.0, marker='*', markersize=10, markevery=20, color='#ff7f0e')
plt.plot(np.cumsum(-df_test["r"]), label=r'$d=0.1$', linewidth=3.0)
plt.plot(np.cumsum(-df_test_d_is_1["r"]), label=r'$d=1$', marker='v', markersize=7, markevery=20, linewidth=3.0, color='#2ca02c')
plt.plot(np.cumsum(-df_test_d_is_2["r"]), label=r'$d=2$', linewidth=3.0, linestyle='dashed', marker='X', markersize=8, markevery=20, color='#9467bd')
plt.xlim(0, 365)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Day', fontsize=20)
plt.ylabel('Cumulative Costs ($)', fontsize=20)
plt.legend(fontsize=20)
ax.text(368, np.around(np.sum(-df_test["r"])+3,2), cpo_r, style='italic', fontsize='x-large',
        bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-df_test_d_is_1["r"]),2), cpo_r_d_is_1, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#2ca02c', 'alpha': 0.5, 'pad': 5})
ax.text(368, np.around(np.sum(-df_test_d_is_2["r"])-6,2), cpo_r_d_is_2, style='italic',  fontsize='x-large',
        bbox={'facecolor': '#9467bd', 'alpha': 0.5, 'pad': 5})
axh = ax.axhline(y=np.sum(-df_test["r"]))
axh.set_linestyle('--')
axh.set_color('#7f7f7f')
axh = ax.axhline(y=np.sum(-df_test_d_is_1["r"]))
axh.set_linestyle('--')
axh.set_color('#7f7f7f')
axh = ax.axhline(y=np.sum(-df_test_d_is_2["r"]))
axh.set_linestyle('--')
axh.set_color('#7f7f7f')
ax.yaxis.set_label_coords(-0.11,0.5)
plt.tight_layout(rect=(0,0,1,1))
plt.show(block=False)

d = 0.1
cpo_v_01 = "{0:.2f}%".format(np.mean(np.maximum(0, df_test["s"].values-d)/d) * 100)
d = 1.0
cpo_v_1 = "{0:.2f}%".format(np.mean(np.maximum(0, df_test_d_is_1["s"].values-d)/d) * 100)
d = 2.0
cpo_v_2 = "{0:.2f}%".format(np.mean(np.maximum(0, df_test_d_is_2["s"].values-d)/d) * 100)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)
plt.plot(np.cumsum(df_test["s"]), label=r'$d=0.1$', linewidth=3.0)
plt.plot(np.cumsum(df_test_d_is_1["s"]), label=r'$d=1$', marker='v', markersize=7, markevery=20, linewidth=3.0, color='#2ca02c')
plt.plot(np.cumsum(df_test_d_is_2["s"]), label=r'$d=2$', marker='X', markersize=8, markevery=20, linewidth=3.0, linestyle='dashed', color='#9467bd')
plt.xlim(0, 365)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Day', fontsize=20)
plt.ylabel('Cumulative Constraint Values (kWh)', fontsize=20)
ax.text(370, 70, cpo_v_01, style='italic', fontsize='x-large',
        bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 7})
ax.text(370, 420, cpo_v_1, style='italic', fontsize='x-large',
        bbox={'facecolor': '#2ca02c', 'alpha': 0.5, 'pad': 7})
ax.text(370, 780, cpo_v_2, style='italic', fontsize='x-large',
        bbox={'facecolor': '#9467bd', 'alpha': 0.5, 'pad': 7})
axh = ax.axhline(y=399)
axh.set_linestyle('--')
axh.set_color('#7f7f7f')
axh1 = ax.axhline(y=780)
axh1.set_linestyle('--')
axh1.set_color('#7f7f7f')
plt.legend(fontsize=20)
plt.tight_layout(rect=(0,0,1,1))
plt.show(block=True)