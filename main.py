from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np

from agents.cem import CEM

# Gym params
EXPERIMENT_DIR = './cartpole-experiment-1'

if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make('CartPole-v0')
    env.monitor.start(EXPERIMENT_DIR, force=True)

    num_features = env.observation_space.shape[0]

    cem = CEM(num_features = num_features, \
        env = env, \
        batch_size = 30, \
        max_num_steps = 200, \
        elite_frac = .2, \
        n_iter = 50)

    perf_hist = cem.train()
    print(perf_hist)
