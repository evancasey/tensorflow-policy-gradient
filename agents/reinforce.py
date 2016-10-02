from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
import tensorflow as tf

from . import (utils, base_agent)

class REINFORCE(base_agent.BaseAgent):
    def __init__(self,
                 env,
                 render,
                 debug,
                 sess,
                 action_policy,
                 num_features,
                 batch_size,
                 max_num_steps,
                 n_iter,
                 algo_discount):

        super(REINFORCE, self).__init__(env, render, debug, sess, action_policy,
                num_features, batch_size, max_num_steps, n_iter)

        # params specific to the policy gradient algo 
        self.algo_discount = algo_discount

        with tf.variable_scope("policy"):
            self.actions = tf.placeholder(tf.int32, [None, 1], "actions")
            self.returns = tf.placeholder(tf.float32, [None, 1], "returns")

            num_actions = self.env.action_space.n
            action_mask = tf.one_hot(indices=self.actions, depth=num_actions)
            # TODO: why are we using softmax here?
            self.log_probs = tf.nn.log_softmax(self.action_policy.network.logits)
            self.policy_probs = tf.reduce_sum( \
                    tf.mul(self.log_probs, action_mask), reduction_indices = 1)
            # negative since we are maximizing 
            self.loss = -tf.reduce_sum(tf.mul(self.policy_probs, utils.standardize(self.returns)))
            self.opt = tf.train.AdamOptimizer(self.action_policy.learning_rate).minimize(self.loss)

    def train(self):
        tf.initialize_all_variables().run()

        for i in xrange(self.n_iter):

            # Step 1: Collect samples with the current policy
            paths = self.obtain_samples(self.render)

            # Step 2: Compute returns on each path
            states, actions, rewards, returns = self.process_samples(paths)

            # Step 3: Compute and apply grad updates
            _, policy_loss = self.sess.run( \
                    [self.opt, self.loss], \
                    feed_dict = {self.action_policy.network.observations: states,
                                 self.actions: actions,
                                 self.returns: returns})

            print("Policy loss: ", float(policy_loss))

            if self.debug: 
                self.print_paths(paths)

    def process_samples(self, paths):
        '''
        Compute return for each path (the discounted reward).

        Returns a list of paths containing returns.
        '''
        
        states, actions, rewards, returns = [], [], [], []
        for path in paths:
            path["returns"] = utils.discount_cum_sum(path["rewards"], \
                    self.algo_discount)
            
            states.append(np.array(path["states"]).reshape(-1, 4))
            actions.append(np.array(path["actions"]).reshape(-1,1))
            rewards.append(np.array(path["rewards"]).reshape(-1, 1))
            returns.append(np.array(path["returns"]).reshape(-1, 1))

        return np.vstack(states), \
               np.vstack(actions), \
               np.vstack(rewards), \
               np.vstack(returns)
