import tensorflow as tf
from baselines.common.models import mlp, lstm

def ev():
    return dict(
        network = mlp(num_hidden=64, num_layers=3),
        timesteps_per_batch=7000,
        max_kl=0.01,
        max_sf=0.1,
        gamma=0.99,
        lam=0.95,
        ent_coef=0.01,
        activation=tf.nn.relu,
        normalize_observations=True,
        value_network='copy',
    )
