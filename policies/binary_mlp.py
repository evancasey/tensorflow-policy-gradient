import tensorflow as tf
import numpy as np

class BinaryMLPPolicy(object):
    def __init__(self, network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate

    def calc_action(self, observation):
        self.debug_softmax = tf.exp(self.network.logits)
        sample_action = tf.multinomial(self.network.logits, num_samples = 1)
        self.sample_action_op = tf.reshape(sample_action, shape = [])
        sao = self.network.sess.run( \
            self.sample_action_op, \
            feed_dict = {self.network.observations: [observation] })

        #print("ds: ", ds)
        #print("sao: ", sao)
        return sao
