class Policy_Base:
    def __init__(self, theta):
        self.theta = theta

    def set_theta(self, theta):
        self.theta = theta

    def get_gradient(self, action, state, theta = None):
        if theta is None:
            return self.get_gradient(theta = self.theta)
        else:
            raise NotImplementedError

    def policy(self, action, state):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError