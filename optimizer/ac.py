import numpy as np
from policy_improvement.softmax import SoftmaxPolicy


class ActorCritic:
    def __init__(self, env, eta_theta=0.01, eta_omega=0.01, gamma=0.95, is_on_policy = True):
        self.features = env.features
        self.num_features = env.num_features
        self.env = env
        self.is_on_policy = is_on_policy

        self.eta_theta = eta_theta
        self.eta_omega = eta_omega
        self.gamma = gamma

        self.theta = None
        self.omega = None
        self.policy = None

        self.theta = np.zeros((self.num_features, 1))

    def set_theta(self, theta):
        self.theta = np.squeeze(theta)
        self.omega = np.zeros_like(theta)
        self.policy = SoftmaxPolicy(theta, self.env, 1.0)

    def set_omega(self, omega):
        self.omega = omega

    def update(self, state, action, num_traj = 30, len_traj = 60):
        if self.is_on_policy:
            IS_ratio = 1.0
        else:
            IS_ratio = self.policy.policy(action, state)/self.env.behavior_policy[state, action]

        grad_ln_pi = self.policy.get_ln_gradient(action, state)

        phi_t = self.env.phi_table(action, state)
        Q_t = np.dot(self.omega, phi_t)

        env = self.env.get_copy()
        env.current_state = state
        next_state, reward, next_action = env.step()
        phi_tt = self.env.phi_table(next_action, next_state)
        Q_tt = np.dot(self.omega, phi_tt)
        delta_t = reward + self.gamma * Q_tt - Q_t

        self.theta = np.squeeze(self.theta + self.eta_theta * IS_ratio * Q_t * np.squeeze(grad_ln_pi))
        self.omega = np.squeeze(self.omega + self.eta_omega * IS_ratio * delta_t * phi_t)

