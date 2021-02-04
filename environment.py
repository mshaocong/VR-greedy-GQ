import numpy as np
import utils
import gym

class FrozenLake:
    def __init__(self):
        state_trans_kernel = np.zeros((16, 16))
        state_trans_kernel[0, (1, 4)] = 0.25
        state_trans_kernel[0, 0] = 0.5
        state_trans_kernel[1, (0, 1, 2, 5)] = 0.25
        state_trans_kernel[2, (1, 2, 3, 6)] = 0.25
        state_trans_kernel[3, (2, 7)] = 0.25
        state_trans_kernel[3, 3] = 0.5
        state_trans_kernel[4, (0, 4, 5, 8)] = 0.25
        state_trans_kernel[5, 0] = 1.0
        state_trans_kernel[6, (2, 5, 7, 10)] = 0.25
        state_trans_kernel[7, 0] = 1.0
        state_trans_kernel[8, (4, 8, 9, 12)] = 0.25
        state_trans_kernel[9, (5, 8, 11, 13)] = 0.25
        state_trans_kernel[10, (6, 9, 11, 14)] = 0.25
        state_trans_kernel[11, 0] = 1.0
        state_trans_kernel[12, 0] = 1.0
        state_trans_kernel[13, (9, 12, 13, 14)] = 0.25
        state_trans_kernel[14, (10, 13, 14, 15)] = 0.25
        state_trans_kernel[15, 0] = 1.0

        reward = np.zeros(16)
        reward[-1] = 1.0

        self.num_state = 16
        self.num_action = 4
        self.num_features = 4

        self.behavior_policy = utils.get_uniform_policy(16, 4)

        self.state_action_trans_kernel = None

        self.trans_kernel = state_trans_kernel
        self.features = utils.get_features(4, 16, 4)

        self.state_space = np.arange(16)
        self.action_space = np.arange(4)
        self.current_state = self.state_space[0]
        self.reward = reward

        self._env = gym.make("FrozenLake-v0", is_slippery=False)
        self._env .reset()

    def reset(self):
        self.current_state = self.state_space[0]

    def phi_table(self, action, state):
        return self.features[action, state, :].reshape((self.num_features, 1))

    def step(self):
        """
        :return: next state, reward, action
        """
        # randomly pick one action based on the current state
        action = np.random.choice(a=self.action_space, p=self.behavior_policy[self.current_state, :])
        # randomly pick the next state

        state, reward, done, info = self._env.step(action)

        next_state = state

        self.current_state = np.copy(next_state)
        return next_state, reward, action

    def get_copy(self):
        env =  FrozenLake()
        env.trans_kernel = np.copy(self.trans_kernel)
        env.features = np.copy(self.features)
        return env