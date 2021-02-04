import numpy as np
import utils


class Garnet:
    def __init__(self, num_state, num_action, branching_factor, num_features):
        self.num_state = num_state
        self.num_action = num_action
        self.branching_factor = branching_factor
        self.num_features = num_features

        self.behavior_policy = utils.get_uniform_policy(num_state, num_action)
        self.state_action_trans_kernel = utils.get_random_state_action_trans_kernel(num_state, num_action)
        self.trans_kernel = np.einsum('iij->ij', self.behavior_policy.dot(self.state_action_trans_kernel))
        self.features = utils.get_features(num_action, num_state, num_features)

        self.state_space = np.arange(num_state)
        self.action_space = np.arange(num_action)
        self.current_state = self.state_space[0]
        self.reward = np.random.uniform(size=num_state)

    def set_behavior_policy(self, policy):
        self.behavior_policy = policy
        self.trans_kernel = np.einsum('iij->ij', self.behavior_policy.dot(self.state_action_trans_kernel))

    def reset(self):
        self.current_state = self.state_space[0]

    def phi_table(self, action, state):
        return self.features[action, state, :].reshape((self.num_features, 1))

    def bellman_operator(self, v_theta, gamma=0.95):
        stationary = np.diag(utils.compute_stationary_dist(self.trans_kernel))
        inv = np.matmul(np.matmul(np.transpose(self.features), stationary), self.features)
        inv = np.linalg.inv(inv)
        projecion = np.matmul(np.matmul(self.features, inv), np.transpose(self.features))
        projecion = np.matmul(projecion, stationary)
        return self.reward.reshape(self.num_state, 1) + gamma*np.matmul(projecion, np.matmul(self.trans_kernel, v_theta))

    def step(self):
        """
        :return: next state, reward, action
        """
        # randomly pick one action based on the current state
        action = np.random.choice(a=self.action_space, p=self.behavior_policy[self.current_state, :])
        # randomly pick the next state
        probs = self.state_action_trans_kernel[self.current_state, action, :]
        next_state = np.random.choice(a=self.state_space, p=probs)
        reward = self.reward[next_state]

        self.current_state = np.copy(next_state)
        return next_state, reward, action

    def sample(self):
        action = np.random.choice(a=self.action_space, p=self.behavior_policy[self.current_state, :])
        # randomly pick the next state
        probs = self.state_action_trans_kernel[self.current_state, action, :]
        next_state = np.random.choice(a=self.state_space, p=probs)
        reward = self.reward[next_state]
        return self.current_state, action, next_state, reward

    def get_copy(self):
        env =  Garnet(self.num_state, self.num_action, self.branching_factor, self.num_features)
        env.set_behavior_policy(policy=self.behavior_policy)
        env.state_action_trans_kernel = np.copy(self.state_action_trans_kernel)
        env.features = np.copy(self.features)
        env.reward = np.copy(self.reward)
        return env