from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
import tensorflow as tf

from agents.cem import CEM
from networks.linear import LinearModel
from networks.softmax_linear import SoftmaxLinearModel

# Gym params
EXPERIMENT_DIR = './cartpole-experiment-1'

if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make('CartPole-v0')
    env.monitor.start(EXPERIMENT_DIR, force=True)

    num_features = env.observation_space.shape[0]

    with tf.Session() as sess:
        linear_network = LinearModel(sess, num_features)
        cem = CEM(num_features = num_features, \
            sess = sess, \
            pred_network = linear_network, \
            env = env, \
            batch_size = 30, \
            max_num_steps = 200, \
            elite_frac = .2, \
            n_iter = 20)

        cem.train()
        print(cem.perf_hist)

        softmax_linear_network = SoftmaxLinearModel(sess, num_features)
        cem = CEM(num_features = num_features, \
            sess = sess, \
            pred_network = softmax_linear_network, \
            env = env, \
            batch_size = 30, \
            max_num_steps = 200, \
            elite_frac = .2, \
            n_iter = 20)

        cem.train()
        print(cem.perf_hist)


