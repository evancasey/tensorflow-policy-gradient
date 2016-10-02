from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
import tensorflow as tf

from . import utils

class VanillaActorCritic(object):
    def __init__(self,
                 env,
                 sess,
                 policy,
                 baseline,
                 num_features,
                 batch_size,
                 max_num_steps,
                 n_iter,
                 algo_discount,
                 gae_lambda):
        self.env = env
        self.sess = sess
        self.policy = policy
        self.baseline = baseline
        self.num_features = num_features
        self.batch_size = batch_size
        self.max_num_steps = max_num_steps
        self.n_iter = n_iter
        self.algo_discount = algo_discount
        self.gae_lambda = gae_lambda
        self.perf_hist = []

        with tf.variable_scope("id"):
            self.opt = tf.group(self.policy.opt, self.baseline.opt)

    def train(self, render = False):
        
        tf.initialize_all_variables().run()

        for i in xrange(self.n_iter):

            # Step 1: Collect samples with the current policy
            paths = self.obtain_samples(render)

            # Step 2: Compute advantage on each path 
            states, actions, advantages, returns = self.process_samples(paths)

            # Step 3: Optimize the value and policy networks jointly
            _, critic_loss, actor_loss = self.sess.run( \
                    [self.opt, self.baseline.loss, self.policy.loss], \
                    feed_dict = {self.policy.network.inputs: states, \
                                 self.baseline.network.inputs: states, \
                                 self.baseline.returns : returns, \
                                 self.policy.actions: actions, \
                                 self.policy.advantages: advantages})

            print("Critic loss: ", critic_loss)
            print("Actor loss: ", actor_loss)

    def obtain_samples(self, render):
        '''
        Performs `batch_size` rollouts.

        Returns a list of paths.
        '''
        paths = []
        for _ in xrange(self.batch_size):
            path = self.rollout(render)
            paths.append(path)

        return paths
        
    def rollout(self, render):
        '''
        Plays one episode to `max_num_steps` or a terminal state using the
        current weights in `pred_network`.

        Returns a path (observation, action, reward).
        '''

        observation = self.env.reset() 
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

        states, actions, rewards = [], [], []
        for _ in xrange(self.max_num_steps):
            states.append(observation)
            
            if np.random.random() < 0.8:
                action = self.policy.calc_action(observation)
            else:
                action = np.random.randint(0, self.env.action_space.n - 1)
            observation, reward, is_terminal, info = self.env.step(action)
            
            one_hot_action = np.zeros(self.env.action_space.n)
            one_hot_action[action] = 1
            actions.append(one_hot_action)
            rewards.append(reward)

            if is_terminal:
                break

            if render:
                self.env.render()

        return {"states": states, \
                "actions": actions, \
                "rewards" : rewards}

    def process_samples(self, paths):
        '''
        Compute advantage for each path (the difference between the future
        discount reward of each action and the baseline's prediction).

        Returns a list of paths containing advantages and returns.
        '''
        

        states, actions, advantages, returns = [], [], [], []
        for path in paths:
            path_baselines = np.append([self.baseline.calc_value(state) for \
                    state in path["states"]], 0)
            deltas = path["rewards"] + \
                     self.algo_discount * path_baselines[1:] - \
                     path_baselines[:-1]

            path["advantages"] = utils.discount_cum_sum(deltas, \
                    self.algo_discount * self.gae_lambda)
            path["returns"] = utils.discount_cum_sum(path["rewards"], \
                    self.algo_discount)
            
            states.append(np.array(path["states"]).reshape(-1, 4))
            actions.append(np.array(path["actions"]).reshape(-1, 2))
            advantages.append(np.array(path["advantages"]).reshape(-1, 1))
            returns.append(np.array(path["returns"]).reshape(-1, 1))

        return np.vstack(states), \
               np.vstack(actions), \
               np.vstack(advantages), \
               np.vstack(returns)
