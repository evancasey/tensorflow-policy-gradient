from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
import tensorflow as tf

from agents.pg import PolicyGradient
from networks.nn import FullyConnectedNN
from policies.binary_mlp import BinaryMLPPolicy
from baselines.mlp import MLPBaseline

# Gym params
EXPERIMENT_DIR = './cartpole-experiment-1'

if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make('CartPole-v0')
    env.monitor.start(EXPERIMENT_DIR, force=True)

    num_features = env.observation_space.shape[0]

    with tf.Session() as sess:
        action_network = FullyConnectedNN(
                sess = sess, 
                net_dims = [num_features, 10, 2])
        value_network = FullyConnectedNN(
                sess = sess, 
                net_dims = [num_features, 10, 2])

        pg = PolicyGradient(env = env, \
            sess = sess, \
            policy = BinaryMLPPolicy(action_network), \
            baseline = MLPBaseline(value_network), \
            num_features = num_features, \
            batch_size = 10, \
            max_num_steps = 200, \
            n_iter = 100, \
            algo_discount = .1, \
            gae_lambda = .1)

        pg.train()
