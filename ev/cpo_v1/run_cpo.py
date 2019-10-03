import os
import sys
import time
sys.path.insert(0, '/home/cisa3/hepeng/Github/baselines/baselines')
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import multiprocessing
import tensorflow as tf, numpy as np
from baselines import logger
from baselines.cpo.cpo import learn
from defaults import ev

import gym
from baselines.bench import Monitor
from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper

dir_name = 'cpo_d_is_2'

def make_env(env_id, seed, train=True, logger_dir=None, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for safety.
    """
    env = gym.make(env_id, **{"train":train})
    env = Monitor(env, logger_dir, allow_early_resets=True, info_keywords=tuple("s"))
    env.seed(seed)
    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

t0 = time.time()
# Train
seed = 1
train = True
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo_v1/train/{}/'.format(dir_name)

env = make_env('EVCharging-v0', seed, train, logger_dir, 10.0)
total_timesteps = 5e7
alg_kwargs = ev()
model = learn(
    env=env,
    seed=seed,
    total_timesteps=total_timesteps,
    **alg_kwargs
)
save_path='/home/lihepeng/Documents/Github/tmp/ev/cpo_v1/train/{}/cpo.ckpt'.format(dir_name)
model.save(save_path)

env.close()
df_train = load_results(logger_dir)
t1 = time.time()
print('Total training time is {}'.format(t1-t0))