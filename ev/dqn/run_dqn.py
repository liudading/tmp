import gym
import numpy as np
from baselines import deepq
from baselines.common.models import mlp
from baselines.common.cmd_util import make_env

penalty = '0.1'
seed = 321
def main():
    env = make_env("EVCharging-v1", "safety", seed=seed, wrapper_kwargs={'frame_stack': True})
    act = deepq.learn(
        env,
        network=mlp(num_hidden=64, num_layers=3),
        lr=1e-3,
        batch_size=64,
        total_timesteps=200000,
        buffer_size=50000,
        exploration_fraction=0.9,
        exploration_final_eps=0.02,
        print_freq=100,
    )
    save_path = '/home/lihepeng/Documents/Github/tmp/ev/dqn/train/dqn_eta_is_{}.pkl'.format(penalty)
    act.save(save_path)

if __name__ == '__main__':
    main()