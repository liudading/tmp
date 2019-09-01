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

# Train
seed = 1
env = make_env(
    env_id='SmartHome-v1',
    seed=seed,
    train=True,
    logger_dir='/home/lihepeng/Documents/Github/tmp/dr/trpo/train',
    reward_scale=1.0,
    )
model = learn(
    env=env,
    seed=seed,
    total_timesteps=1e8,
    **dr(),
    )
model.save('/home/lihepeng/Documents/Github/tmp/dr/trpo/train/trpo.ckpt')
env.close()