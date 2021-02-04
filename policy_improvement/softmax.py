from policy_improvement.policybase import Policy_Base
import numpy as np

class SoftmaxPolicy(Policy_Base):
    def __init__(self, theta, env, sigma):
        super().__init__(theta)
        self.sigma = sigma
        self.env = env

    def get_gradient(self, action, state, theta = None):
        if theta is None:
            return self.get_gradient(action, state, theta = self.theta)
        else:
            phi = self.env.phi_table(action, state)
            numerator = np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi))
            dominator = 0.0
            dominator2 = np.zeros_like(phi)
            for a in self.env.action_space:
                phi_pine = self.env.phi_table(a, state)
                dominator = dominator + np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi_pine))
                dominator2 = dominator2 + self.sigma*numerator* np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi_pine)) * phi_pine
            return (self.sigma * numerator * phi - dominator2)/dominator**2

    def get_ln_gradient(self, action, state, theta = None):
        if theta is None:
            return self.get_ln_gradient(action, state, theta = self.theta)
        else:
            phi = self.env.phi_table(action, state)
            # numerator = np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi))
            dominator = 0.0
            dominator2 = np.zeros_like(phi)
            for a in self.env.action_space:
                phi_pine = self.env.phi_table(a, state)
                dominator = dominator + np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi_pine))
                dominator2 = dominator2 + self.sigma* np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi_pine)) * phi_pine
            return self.sigma * phi - dominator2/dominator

    def policy(self, action, state):
        phi = self.env.phi_table(action, state)
        numerator = np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi))
        dominator = 0.0
        for a in self.env.action_space:
            phi_pine = self.env.phi_table(a, state)
            dominator = dominator + np.exp(self.sigma * np.dot(np.squeeze(np.asarray(self.theta)), phi_pine))
        return numerator/dominator

    def get_action(self, state):
        dist = [self.policy(a, state)[0] for a in self.env.action_space]
        return np.random.choice(self.env.action_space, size=None, p=np.squeeze(dist))