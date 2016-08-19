import tensorflow as tf
import numpy as np

class BinaryMLPPolicy(object):
    def __init__(self,
                 network):
        self.network = network

        with tf.variable_scope("id"):
            self.advantages = tf.placeholder("float", [None, 1], "advantages")
            self.actions = tf.placeholder("float", [None, 2], "actions")

            # reinforce action probs that performed well
            self.log_probs = tf.log(self.network.layers[-1])
            self.policy_probs = tf.reduce_sum( \
                    tf.mul(self.log_probs, self.actions), reduction_indices = 1)
            self.loss = -tf.reduce_sum(tf.mul(self.policy_probs, self.advantages))
            self.opt = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def calc_action(self, observation):
        probs = self.network.predict(observation)[0]
        return self._weighted_choose_action(probs)

    def _weighted_choose_action(self, pi_probs):
        r = np.random.uniform(0, sum(pi_probs))
        upto = 0
        for idx, prob in enumerate(pi_probs):
            if upto + prob >= r:
                return idx
            upto += prob
        return len(pi_probs) - 1

    def fit(self, observations, actions, advantages):
        '''
        Update our policy network in the direction of the score function gradient 
        est. ( grad. log prob(pi) * advantage)
        '''

        import ipdb; ipdb.set_trace()

    
    def weighted_choose_action(self, pi_probs):
        r = random.uniform(0, sum(pi_probs))
        upto = 0
        for idx, prob in enumerate(pi_probs):
            if upto + prob >= r:
                return idx
            upto += prob
        return len(pi_probs) - 1

