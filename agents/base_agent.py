from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
import tensorflow as tf
import json
import collections

class BaseAgent(object):

    def __init__(self,
                 env,
                 render,
                 debug,
                 sess,
                 action_policy,
                 num_features,
                 batch_size,
                 max_num_steps,
                 n_iter):

        self.env = env
        self.render = render
        self.debug = debug
        self.sess = sess
        self.action_policy = action_policy
        self.num_features = num_features
        self.batch_size = batch_size
        self.max_num_steps = max_num_steps
        self.n_iter = n_iter

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
            
            if np.random.random() < 0.9:
                action = self.action_policy.calc_action(observation)
            else:
                action = self.env.action_space.sample()
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

    def print_paths(self, paths):
        for path in paths:
            print("states: ", np.array(path["states"]).tolist())
            print("actions: ", np.array(path["actions"]).tolist())
            print("rewards: ", np.array(path["rewards"]).tolist())
        print("\n")


