import gym
import numpy as np
from baselines import deepq
from baselines.common.models import mlp
from baselines.common.cmd_util import make_env

penalty = '0.1'
seed = 1314
def main():
    returns, safeties = [], []
    env = make_env("EVCharging-v1", "safety", seed=seed, wrapper_kwargs={'frame_stack': True}, env_kwargs={'train': False})
    act = deepq.learn(
        env,
        network=mlp(num_hidden=64, num_layers=3),
        lr=1e-3,
        total_timesteps=0,
        load_path='/home/lihepeng/Documents/Github/tmp/ev/dqn/train/dqn_eta_is_{}.pkl'.format(penalty),
    )
    dates = env.unwrapped._price['date'].unique()[1:-1]
    d = 0
    obs, done = env.reset(**{"arr_date": dates[d]}), False
    while True:
        episode_rew, episode_sft = 0, 0
        while not done:
            obs, rew, done, info = env.step(act(obs[None])[0])
            episode_rew += info["r"]
            episode_sft += info["s"]
        print("Episode reward {}, safety {}".format(episode_rew, episode_sft))
        returns.append(episode_rew)
        safeties.append(episode_sft)
        d += 1
        if d >= dates.size:
            break
        # env.render()
        obs, done = env.reset(**{"arr_date": dates[d]}), False

    print('test returns: {}'.format(np.sum(returns)))
    print('test safeties: {}'.format(np.sum(safeties)))

    np.save('/home/lihepeng/Documents/Github/tmp/ev/dqn/test/returns_{}'.format(penalty), returns)
    np.save('/home/lihepeng/Documents/Github/tmp/ev/dqn/test/safeties_{}'.format(penalty), safeties)

if __name__ == '__main__':
    main()