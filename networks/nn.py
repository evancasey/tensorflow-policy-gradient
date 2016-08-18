import tensorflow as tf
import numpy as np

class FullyConnectedNN(object):
    def __init__(self,
                 sess,
                 net_dims,
                 name = "FullyConnected"):

        self.sess = sess
        self.net_dims = net_dims
        self.name = name

        with tf.variable_scope(name):

            # construct input, weights, and biases
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.net_dims[0]], name="inputs")
            self.w = [tf.Variable(tf.random_normal([self.net_dims[i], \
                self.net_dims[i+1]]), name="weights") for i in range(len(self.net_dims) - 1)]
            self.b = [tf.Variable(tf.random_normal([self.net_dims[i]]), \
                name="bias") for i in range(1, len(self.net_dims))]

            assert len(self.w) == len(self.b)
            assert len(self.net_dims) == len(self.w) + 1 

            # construct the layers
            self.layers = [self.inputs]
            for i in range(len(self.w)):
                prev_layer, prev_weights, prev_biases = self.layers[i], self.w[i], self.b[i]
                if i == len(self.w) - 1:
                    self.layers.append(tf.add(tf.matmul(prev_layer, prev_weights), prev_biases))
                else:
                    self.layers.append(tf.nn.relu(tf.add(tf.matmul(prev_layer, prev_weights), prev_biases)))

            assert len(self.layers) == len(net_dims)

    def predict(self, obs):
        '''
        Do forward prop and return the values of the output layer for a single
        observation vector
        '''

        return self.sess.run(self.layers[-1], feed_dict = {self.inputs: \
            obs.reshape(1, self.net_dims[0]) })
