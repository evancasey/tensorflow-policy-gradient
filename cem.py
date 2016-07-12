import gym
import numpy as np
import matplotlib.pyplot as plt

# Gym params
EXPERIMENT_DIR = './cartpole-experiment-1'

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
                 num_features, 
                 batch_size, 
                 max_num_steps,
                 elite_frac, 
                 n_iter):
        
        self.num_features = num_features
        self.batch_size = batch_size
        self.max_num_steps = max_num_steps
        self.elite_frac = elite_frac
        self.n_iter = n_iter

    def train(self):
        """
        Given an initial distribution vector of w_i, compute a new distribution
        vector via the cross-entropy method.

        Returns a policy of type state => action.
        """
        
        # Additional param for bias
        distrib_means = np.zeros(num_features + 1)
        distrib_vars = np.ones(num_features + 1)

        for _ in xrange(self.n_iter):
            # Step 1: sample 'batch_size' w_i's from initial distribution
            batch_weights = self._sample_weights(self.batch_size, distrib_means,
                    distrib_vars)

            # Step 2: perform rollout and evaluate each w_i (eg. call score_policy)
            batch_scores = np.apply_along_axis(self.rollout, 1, batch_weights)

            # Step 3: select the top 'elite_frac' w_i's 
            top_weights = self._top_weights(batch_weights, batch_scores)

            # Step 4: fit a new Gaussian distrib. over the top scoring w_i's
            distrib_means = np.mean(top_weights, 0)
            distrib_vars = np.var(top_weights, 0)

            print distrib_means
            print distrib_vars

        import ipdb; ipdb.set_trace()
        pass

    def rollout(self, w):
        """
        Plays one episode to `max_num_steps` or a terminal state, given a weight vector w. 

        Returns a scalar of the reward sum of the episode.
        """

        theta = w
        observation = env.reset() 
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        total_reward = 0.0
        for _ in xrange(self.max_num_steps -1):
            policy = BinaryLinearModel(theta[:-1], theta[-1])
            action = policy.act(observation)
            observation, reward, is_terminal, info = env.step(action)

            total_reward += reward

            if is_terminal:
                break

        return total_reward

    def _sample_weights(self, batch_size, distrib_means, distrib_vars):
        """
        Samples `batch_size` weight vectors from the gaussian distribution 
        parameterized by distrib_means[i] and distrib_vars[i], where i is the
        column index of the weight vector.

        Returns a 2d array of weight vectors, where each dimension is sampled from `distrib_means` and `distrib_vars`
        """
        distrib_cov = np.diag(distrib_vars)
        return np.random.multivariate_normal(distrib_means, distrib_cov, batch_size)

    def _top_weights(self, batch_weights, batch_scores):
        """
        Selects the `elite_frac` top performing weight vectors.

        Returns a 2d array of the top performing weight vectors.
        """

        # Join weights with their resp. scores, then sorts along the last axis
        weights_with_score = np.hstack((batch_weights, np.array([batch_scores]).T))
        sorted_weights_with_score = np.sort(weights_with_score)

        top_indices = int(self.batch_size * self.elite_frac)
        return sorted_weights_with_score[:top_indices,:-1]

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.monitor.start(EXPERIMENT_DIR, force=True)

    num_features = env.observation_space.shape[0]

    cem = CEM(num_features = num_features, \
        batch_size = 10, \
        max_num_steps = 200, \
        elite_frac = .2, \
        n_iter = 10)

    cem.train()


