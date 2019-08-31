import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('SmartHome-v1')

fig = plt.figure(figsize=(7,3))
plt.plot(range(21), env.price[210:231], color='#7f7f7f', linewidth=2, alpha=0.3)
plt.plot(range(20,164), env.price[230:374], color='r', linewidth=2, label='Electricity price')
plt.xlim(0,163)
plt.xticks([20,163], labels=['$t-T+1$','$t$'], fontsize=20)
plt.yticks(fontsize='large')
plt.ylabel('Price ($)', fontsize='xx-large')
plt.annotate('', xy=(20, 0.015), xytext=(20, 0.0223),
             annotation_clip=False,
             arrowprops=dict(facecolor='#ff7f0e', arrowstyle='-', alpha=1.0))
plt.tight_layout()
plt.legend(fontsize='xx-large')
plt.show()

fig = plt.figure(figsize=(7,3))
plt.plot(range(21), env.temp[210:231], color='#7f7f7f', linewidth=2, alpha=0.3)
plt.plot(range(20,164), env.temp[230:374], color='m', linewidth=2, label='Outdoor temperature')
plt.xlim(0,163)
plt.xticks([20,163], labels=['$t-T+1$','$t$'], fontsize=20)
plt.yticks(range(70,95,5),labels=['70.0','75.0','80.0','85.0','90.0'], fontsize='large')
plt.ylabel('Temperature ($\circ$F)', fontsize='xx-large')
plt.annotate('', xy=(20.2, 67), xytext=(20.2, 70.5),
             annotation_clip=False,
             arrowprops=dict(facecolor='#ff7f0e', arrowstyle='-', alpha=1.0))
plt.tight_layout()
plt.legend(fontsize='xx-large')
plt.show()