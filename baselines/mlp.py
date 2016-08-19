from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

class MLPBaseline(object):
    def __init__(self,
                 network):

        self.network = network
        
        with tf.variable_scope("id"):
            self.returns = tf.placeholder("float", [None, 1], name = "returns")
            self.loss = tf.nn.l2_loss(self.network.layers[-1] - self.returns)
            self.opt = tf.train.AdamOptimizer(0.1).minimize(self.loss)

    def calc_value(self, obs):
        return self.network.predict(obs)[0][0]
    
    # deprecated
    def fit(self, observations, returns):
        _, loss = self.network.sess.run([self.opt, self.loss], \
                feed_dict = {self.network.inputs: observations, \
                             self.returns: returns })

        print("Baseline loss: ", loss)
        


