import gym
import numpy as np
import tensorflow as tf

from agents.reinforce import REINFORCE
from networks.nn import FullyConnectedNN
from policies.binary_mlp import BinaryMLPPolicy

# Gym params
EXPERIMENT_DIR = './cartpole-experiment-1'

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(1234)
    env = gym.make('CartPole-v0')
    env.monitor.start(EXPERIMENT_DIR, force=True)

    num_features = env.observation_space.shape[0]

    with tf.Session() as sess:
        action_network = FullyConnectedNN(
                sess = sess, 
                env = env,
                hidden_layers = "10")

        pg = REINFORCE(env = env, \
            render = True, \
            debug = False, \
            sess = sess, \
            action_policy = BinaryMLPPolicy(action_network, 0.01), \
            num_features = num_features, \
            batch_size = 30, \
            max_num_steps = 200, \
            n_iter = 100, \
            algo_discount = .97)

        pg.train()
