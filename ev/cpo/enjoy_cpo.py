import os
import sys
sys.path.insert(0, '/home/lihepeng/Documents/Github/baselines/baselines')
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import pickle
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
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

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

# Train
seed = 1314
train = False
logger.log("Running trained model")
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/cpo/test'
env = make_env('EVCharging-v0', seed, train, logger_dir, 1.0)
alg_kwargs = ev()
model = learn(
    env=env,
    seed=seed,
    total_timesteps=0,
    load_path='/home/lihepeng/Documents/Github/tmp/ev/cpo/train/cpo.ckpt',
    **alg_kwargs
)

df = env.unwrapped._price
df["soc"] = 0.0
df["act"] = 0.0

d = 0
dates = df['date'].unique()[1:-1]
obs = env.reset(**{"arr_date": dates[d]})
df.loc[env.unwrapped._cur_time, "soc"] = env.unwrapped._soc
while True:
    actions, *_ = model.step(obs)
    obs, rew, done, _ = env.step(actions)
    df.loc[env.unwrapped._cur_time, "act"] = env.unwrapped._act
    df.loc[env.unwrapped._cur_time, "soc"] = env.unwrapped._soc
    if done:
        d += 1
        if d >= dates.size:
            break
        obs = env.reset(**{"arr_date": dates[d]})
        df.loc[env.unwrapped._cur_time, "soc"] = env.unwrapped._soc
env.close()
df.to_csv('/home/lihepeng/Documents/Github/tmp/ev/cpo/test/results.csv')