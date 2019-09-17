import os
import sys
sys.path.insert(0, '/home/lihepeng/Documents/Github/baselines/baselines')
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf, numpy as np
from baselines import logger
from baselines.cpo.cpo import learn
from defaults import dr

import gym
from baselines.bench import Monitor
from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers, set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        low, high = self.env.unwrapped._feasible_action()

        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, low, high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_env(env_id, seed, train=True, logger_dir=None, reward_scale=1.0, mpi_rank=0, subrank=0):
    """
    Create a wrapped, monitored gym.Env for safety.
    """
    env = gym.make(env_id, **{"train":train})
    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env, 
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    env.seed(seed)
    env = ClipActionsWrapper(env)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_vec_env(env_id, seed, train=True, logger_dir=None, reward_scale=1.0, num_env=1.0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id,
            seed,
            train=True,
            logger_dir=None,
            reward_scale=reward_scale,
            mpi_rank=mpi_rank,
            subrank=0
        )
    set_global_seeds(seed)
    return DummyVecEnv([make_thunk(i) for i in range(num_env)])

# Test
seed = 1314
env = make_env(
    env_id='SmartHome-v1',
    seed=seed,
    train=False,
    logger_dir='/home/lihepeng/Documents/Github/tmp/dr/trpo/test',
    reward_scale=1.0,
    )
model = learn(
    env=env,
    seed=seed,
    total_timesteps=0,
    load_path='/home/lihepeng/Documents/Github/tmp/dr/trpo/train/trpo.ckpt',
    **dr(),
    )

keys = ["Price","SoC","WtTemp","flow","IdTemp","OdTemp",
        "P_ov","P_dw","P_wm","P_cd",
        "P_ev","P_wh","P_ac",
        "P_fg","P_vc","P_hd","P_tv","P_nb","P_lg",
        "A_ov","A_dw","A_wm","A_cd","A_ev",
        "B_ov","B_dw","B_wm","B_cd","B_ev",
]
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
values=[deque(maxlen=env.unwrapped.T) for _ in range(len(keys))]
df_monitor = pd.DataFrame(dict(zip(keys, values)))
f_vals = []
for d in range(61):
    returns = 0.0
    obs = env.reset(**{"day": d})
    while True:
        actions, *_ = model.step(obs)
        obs, rew, done, info = env.step(actions)
        returns += info["C_elec"]
        if done:
            df_monitor.append(env.unwrapped.monitor)
            f_vals.append(returns)

env.close()
df_monitor.to_csv('/home/lihepeng/Documents/Github/tmp/dr/trpo/test/results.csv')

print(np.mean(f_vals))
plt.plot(f_vals)
plt.show()
np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/trpo/test/returns.txt', f_vals)
