import tensorflow as tf
from baselines.common.models import mlp, lstm

def dr():
    return dict(
        network = mlp(num_hidden=128, num_layers=3),
        timesteps_per_batch=144*100,
        max_kl=0.01,
        gamma=0.995,
        lam=0.95,
        ent_coef=0.01,
        activation=tf.nn.relu,
        normalize_observations=True,
        value_network='copy',
    )
