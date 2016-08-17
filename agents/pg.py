from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
import tensorflow as tf

from . import utils

class PolicyGradient(object):
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

    def train(self, render = False):
        
        tf.initialize_all_variables().run()

        for i in xrange(self.n_iter):

            # Step 1: Collect samples with the current policy
            paths = self.obtain_samples(render)

            # Step 2: Compute advantage on each path and train our baseline
            processed_paths = self.process_samples(paths)

            # Step 3: Update our policy network

            pass

    def obtain_samples(self, render):
        '''
        Performs `batch_size` rollouts.

        Returns a list of paths.
        '''
        paths = []
        for _ in xrange(self.batch_size - 1):
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
        for _ in xrange(self.max_num_steps - 1):
            states.append(observation)

            prob = self.policy.predict(observation.reshape(1,4))[0][0]
            action = 0 if np.random.uniform(0,1) < prob else 1

            observation, reward, is_terminal, info = self.env.step(action)

            actions.append(action)
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
        Compute advantage for each path (the difference between the future reward of each action and the baseline's prediction) and fit the baseline network with the discounted reward.

        Returns a list of paths containing advantages.
        '''
        
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append([self.baseline.predict(state.reshape(1,4)) for state in
                path["states"]], 0)
            deltas = path["rewards"] + \
                    self.algo_discount * path_baselines[1:] - \
                     path_baselines[:-1]

            path["advantages"] = utils.discount_cum_sum(deltas,
                    self.algo_discount * self.gae_lambda)
            path["returns"] = utils.discount_cum_sum(path["rewards"],
                    self.algo_discount)

            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        # TODO: update value network 
        return paths


    def optimize_policy(self, processed_paths):
        '''
        Update our policy network in the direction of the score function gradient est. ( grad. log prob(pi) * advantage)
        '''
        pass
                    



