import numpy as np
from policy_improvement.softmax import SoftmaxPolicy


class PolicyGradient:
    def __init__(self, env, eta_theta=0.01, gamma=0.95, is_on_policy = True):
        self.features = env.features
        self.num_features = env.num_features
        self.env = env
        self.is_on_policy = is_on_policy

        self.eta_theta = eta_theta
        self.gamma = gamma

        self.theta = None
        self.policy = None

        self.theta = np.zeros((self.num_features, 1))

    def set_theta(self, theta):
        self.theta = theta
        self.policy = SoftmaxPolicy(theta, self.env, 1.0)

    def update(self, state, action, num_traj = 30, len_traj = 60):
        grad_ln_pi = self.policy.get_ln_gradient(action, state)

        G_t = 0.0
        for i in range(num_traj):
            G_tmp = 0.0
            env = self.env.get_copy()
            current_state = self.env.current_state
            env.current_state = current_state
            for j in range(len_traj):
                next_state, reward, action = env.step()
                G_tmp = G_tmp  + self.gamma**i * reward
            G_tmp = G_tmp/len_traj
            G_t += G_tmp
        G_t = G_t / num_traj

        if self.is_on_policy:
            IS_ratio = 1.0
        else:
            IS_ratio = self.policy.policy(action, state)/self.env.behavior_policy[state, action]
        self.theta = self.theta + self.eta_theta * IS_ratio * G_t * np.squeeze(grad_ln_pi)