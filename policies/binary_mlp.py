import tensorflow as tf
import numpy as np

class BinaryMLPPolicy(object):
    def __init__(self,
                 network):
        self.network = network

        with tf.variable_scope(network.name):
            # construct optimizer!
            pass

    def calc_action(self, observation):
        prob = self.network.predict(observation)[0][0]
        return 0 if np.random.uniform(0,1) < prob else 1

