import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim

class FullyConnectedNN(object):
    def __init__(self,
                 sess,
                 env,
                 hidden_layers):

        self.sess = sess
        self.env = env

        with tf.variable_scope("model"):

            num_actions = self.env.action_space.n
            shape_with_batch = [None] + list(self.env.observation_space.shape)
            self.observations = tf.placeholder(shape=shape_with_batch,
                                   dtype=tf.float32)

            flat_input_state = slim.flatten(self.observations, scope='flat')
            final_hidden = self.hidden_layers_starting_at(flat_input_state, hidden_layers)
            self.logits = slim.fully_connected(inputs=final_hidden,
                            num_outputs=num_actions,
                            activation_fn=None)

    def hidden_layers_starting_at(self, layer, config):
        layer_sizes = map(int, config.split(","))
        assert len(layer_sizes) > 0
        for i, size in enumerate(layer_sizes):
            layer = slim.fully_connected(scope="h%d" % i,
                              inputs=layer,
                              num_outputs=size,
                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                              activation_fn=tf.nn.relu)
        return layer
 

    def predict(self, obs):
        '''
        Do forward prop and return the values of the output layer for a single
        observation vector
        '''

        return self.sess.run(self.layers[-1], feed_dict = {self.inputs: \
            obs.reshape(1, self.net_dims[0]) })
