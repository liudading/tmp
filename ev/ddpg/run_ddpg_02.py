import os
import sys
sys.path.insert(0, '/home/lihepeng/Documents/Github/baselines/baselines')
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import multiprocessing
import tensorflow as tf, numpy as np
from baselines import logger
from baselines.ddpg.ddpg import learn
from baselines.common.models import mlp

import gym
from baselines.bench import Monitor
from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

penalty = 0.2

def make_env(env_id, seed, train=True, logger_dir=None, mpi_rank=0, subrank=0, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for safety.
    """
    env = gym.make(env_id, **{"train":train, "penalty":penalty})
    env = Monitor(env, 
        logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
        allow_early_resets=True)
    env.seed(seed)
    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

nenv = ncpu = multiprocessing.cpu_count()
mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
def make_thunk(rank, seed, train, logger_dir, mpi_rank):
    return lambda: make_env(
        env_id='EVCharging-v2',
        seed=seed,
        train=train,
        logger_dir=logger_dir,
        mpi_rank=mpi_rank,
        subrank=rank,
        reward_scale=1.0,
    )

# Train
seed = 321
train = True
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/train'
env = DummyVecEnv([make_thunk(i, seed, train, logger_dir, mpi_rank) for i in range(nenv)])

model = learn(
    network='mlp',
    num_hidden=64,
    num_layers=3,
    env=env,
    seed=seed,
    total_timesteps=500000,
    nb_eval_steps=2000,
)

env.close()
df_train = load_results(logger_dir)

# Test
seed = 1314
train = False
logger.log("Running trained model")
logger_dir = '/home/lihepeng/Documents/Github/tmp/ev/ddpg/test'
env = DummyVecEnv([make_thunk(i, seed, train, logger_dir, mpi_rank) for i in range(nenv)])
dates = env.envs[0].unwrapped._price['DATE'].unique()[1:-1]

d = 0
obs = env.envs[0].unwrapped.reset(**{"arr_date": dates[d]})

returns, safeties = [], []
episode_rew, episode_sft = 0, 0
while True:
    actions, *_ = model.step(obs)
    obs, rew, done, info = env.envs[0].unwrapped.step(actions)
    episode_rew += info["r"]
    episode_sft += info["s"]
    if done:
        returns.append(episode_rew)
        safeties.append(episode_sft)
        episode_rew, episode_sft = 0, 0
        d += 1
        if d >= dates.size:
            break
        obs = env.envs[0].unwrapped.reset(**{"arr_date": dates[d]})

print('test returns: {}'.format(np.sum(returns)))
print('test safeties: {}'.format(np.sum(safeties)))

np.save('/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/returns_{}'.format(str(penalty)), returns)
np.save('/home/lihepeng/Documents/Github/tmp/ev/ddpg/test/safeties_{}'.format(str(penalty)), safeties)