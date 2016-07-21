import gym
import numpy as np

from network import BinearLinearModel

class Agent(object):
    def __init__(self, 
                 env,
                 pred_network,
                 max_num_steps):
        
        self.network = BinaryLinearModel()

    def do_rollout(self, w):
        """
        Simulate the env and agent for `max_num_steps` given a single weight vector `w`
        Returns: [(obs,action,reward)] where obs is the state before action occurs, reward results from action
        """

        observation = env.reset()
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

        for step in range(max_num_steps - 1):
            observation, reward, is_terminal, info = env.step(action)

        pass
