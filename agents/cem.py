from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np


class BinaryLinearModel(object):
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def act(self, x):
        """
        Predicts the output (action) based on the linear model and the input
        state `x`.

        Returns a binary action (0,1).
        """
        y = x.dot(self.w) + self.b
        a = int(y < 0)
        return a

class CEM(object):
    def __init__(self,
                 env,
                 num_features, 
                 batch_size, 
                 max_num_steps,
                 elite_frac, 
                 n_iter):
        
        self.env = env
        self.num_features = num_features
        self.batch_size = batch_size
        self.max_num_steps = max_num_steps
        self.elite_frac = elite_frac
        self.n_iter = n_iter

    def train(self, render = False):
        """
        Given an initial distribution vector of w_i, compute a new distribution
        vector via the cross-entropy method.

        Returns a policy of type state => action.
        """
        
        # Additional param for bias
        distrib_means = np.zeros(self.num_features + 1)
        distrib_vars = np.full((self.num_features + 1,), .1)

        perf_hist = []
        for i in xrange(self.n_iter):

            # Step 1: sample 'batch_size' w_i's from initial distribution
            batch_weights = self._sample_weights(self.batch_size, distrib_means,
                    distrib_vars)

            # Step 2: perform rollout and evaluate each w_i 
            batch_scores = np.apply_along_axis(self.rollout, 1, batch_weights,
                    render)

            perf_hist.append(np.mean(batch_scores, 0))

            # Step 3: select the top 'elite_frac' w_i's 
            top_weights = self._top_weights(batch_weights, batch_scores)

            # Step 4: fit a new Gaussian distrib. over the top scoring w_i's
            noise = max(5 - i / 10, 0)
            distrib_means = np.mean(top_weights, 0)
            distrib_vars = np.var(top_weights, 0) + noise

        return perf_hist

    def rollout(self, w, render):
        """
        Plays one episode to `max_num_steps` or a terminal state, given a weight vector w. 

        Returns a scalar of the reward sum of the episode.
        """

        observation = self.env.reset() 
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

        total_reward = 0.0
        for _ in xrange(self.max_num_steps - 1):
            policy = BinaryLinearModel(w[:-1], w[-1])
            action = policy.act(observation)
            observation, reward, is_terminal, info = self.env.step(action)

            total_reward += reward

            if is_terminal:
                break

            if render:
                self.env.render()

        return total_reward

    def _sample_weights(self, batch_size, distrib_means, distrib_vars):
        """
        Samples `batch_size` weight vectors from the gaussian distribution 
        parameterized by distrib_means[i] and distrib_vars[i], where i is the
        column index of the weight vector.

        Returns a 2d array of weight vectors, where each dimension is sampled from 
        `distrib_means` and `distrib_vars`
        """

        distrib_cov = np.diag(np.sqrt(distrib_vars))
        return np.random.multivariate_normal(distrib_means, distrib_cov, batch_size)

    def _top_weights(self, batch_weights, batch_scores):
        """
        Selects the `elite_frac` top performing weight vectors.

        Returns a 2d array of the top performing weight vectors.
        """

        n_elite = int(np.round(self.batch_size * self.elite_frac))
        elite_inds = batch_scores.argsort()[::-1][:n_elite]
        return batch_weights[elite_inds]





