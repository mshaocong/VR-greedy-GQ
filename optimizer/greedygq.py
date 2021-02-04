import numpy as np
from policy_improvement.softmax import SoftmaxPolicy

class GreedyGQ_Base:
    def __init__(self, env, target_policy, eta_theta=0.01, eta_omega=0.01, gamma=0.95):
        self.features = env.features
        self.num_features = env.num_features
        self.env = env
        
        self.eta_theta = eta_theta
        self.eta_omega = eta_omega
        self.gamma = gamma
        self.target_policy = target_policy

        self.theta = np.zeros((self.num_features, 1))
        self.omega = np.zeros((self.num_features, 1))

    def set_theta(self, theta):
        self.theta = theta
    def set_omega(self, omega):
        self.omega = omega

    def _phi_table(self, action, state):
        return self.features[action, state, :].reshape((self.num_features, 1))

    def _extract_grad_info(self, current_state, reward, next_state, action):
        """
        Input: x=(s,r,s',a); used to compute the pseudo-gradient
        Return: G_x and H_x
        """
        env = self.env
        pi_theta = SoftmaxPolicy(self.theta, env, 1.0)
        theta = self.theta
        omega = self.omega
        gamma = self.gamma
        phi_t = self.env.phi_table(action, current_state)

        # Compute V-bar and phi_hat
        V_bar = 0
        phi_hat = np.zeros_like(phi_t)
        for a in self.env.action_space:
            phi_pine = self.env.phi_table(a, next_state)
            pi_value = pi_theta.policy(a, next_state)
            pi_grad = pi_theta.get_gradient(a, next_state, theta)
            V_bar = V_bar +  pi_value * np.dot(theta, phi_pine)
            phi_hat = phi_hat + np.dot(theta, phi_pine) * pi_grad + pi_value * phi_pine
        delta = reward + gamma * V_bar - np.dot(theta, phi_t)

        G_x = delta * phi_t - gamma * np.dot(omega.transpose(), phi_t) * phi_hat
        H_x = (delta - np.dot(omega.transpose(), phi_t)) * phi_hat
        return  G_x.transpose(), H_x.transpose()

    def _extract_grad_info_2(self, theta, omega, current_state, reward, next_state, action):
        """
        Input: x=(s,r,s',a); used to compute the pseudo-gradient
        Return: G_x and H_x
        """
        env = self.env
        pi_theta = SoftmaxPolicy(theta, env, 1.0)
        gamma = self.gamma
        phi_t = self.env.phi_table(action, current_state)

        # Compute V-bar and phi_hat
        V_bar = 0
        phi_hat = np.zeros_like(phi_t)
        for a in self.env.action_space:
            phi_pine = self.env.phi_table(a, next_state)
            pi_value = pi_theta.policy(a, next_state)
            pi_grad = pi_theta.get_gradient(a, next_state, theta)
            V_bar = V_bar +  pi_value * np.dot(np.squeeze(np.asarray(theta)), phi_pine)
            phi_hat = phi_hat + np.dot(np.squeeze(np.asarray(theta)), phi_pine) * pi_grad + pi_value * phi_pine
        delta = reward + gamma * V_bar - np.dot(np.squeeze(np.asarray(theta)), phi_t)

        G_x = delta * phi_t - gamma * np.dot(omega.transpose(), phi_t) * phi_hat
        H_x = (delta - np.dot(omega.transpose(), phi_t)) * phi_hat
        return  G_x.transpose(), H_x.transpose()

    def update(self, current_state, reward, next_state, action):
        G_x, H_x = self._extract_grad_info(current_state, reward, next_state, action)
        self.theta = self.theta + self.eta_theta * G_x
        self.omega = self.omega + self.eta_omega * H_x

        ratio = 1.0
        if np.sum(self.theta ** 2) >= ratio ** 2:
            self.theta = ratio * self.theta / np.sqrt(np.sum(self.theta ** 2))
        return G_x, H_x