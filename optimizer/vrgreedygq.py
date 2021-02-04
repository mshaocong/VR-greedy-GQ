import optimizer.greedygq
import numpy as np
import copy


class VRGreedyGQ(optimizer.greedygq.GreedyGQ_Base):
    def __init__(self, env, batch_size=1000, eta_theta=0.01, eta_omega=0.01, target_policy=None, gamma=0.95):
        super(VRGreedyGQ, self).__init__(env, target_policy, eta_theta=eta_theta,  eta_omega=eta_omega, gamma=0.95)
        self.batch_size = batch_size

        self._grad_cache = {"G": np.zeros((1, self.env.num_features)),
                            "H": np.zeros((1, self.env.num_features)) }
        self.batch_grad_info = None

        self._pars_cache = {"theta": np.copy(self.theta), "omega": np.copy(self.omega)}
        self.theta_tilde = np.asarray(np.copy(self.theta))
        self.omega_tilde = np.asarray(np.copy(self.omega))

        self._trajectory_cache = []
        self._trajectory = None

    def _get_batch_grad_info(self):
        G = self.batch_grad_info["G"]
        H = self.batch_grad_info["H"]
        return G, H

    def update(self, current_state, reward, next_state, action=None):
        G_x, H_x = self._extract_grad_info_2(self.theta_tilde, self.omega_tilde, current_state, reward, next_state,
                                                 action)

        if len(self._trajectory_cache) < self.batch_size:
            # 开始预存这个trajectory里的full gradient info
            self._trajectory_cache.append((current_state, reward, next_state, action))
            self._grad_cache["G"] = self._grad_cache["G"] + G_x / self.batch_size
            self._grad_cache["H"] = self._grad_cache["H"] + H_x / self.batch_size
        else:
            # 记录了刚好M个样本的时候. 进入下一个batch
            # cache全部重置, theta-tilde更新成前一个batch的均值, theta重设为theta-tilde
            self._trajectory = copy.deepcopy(self._trajectory_cache)
            self._trajectory_cache = [(current_state, reward, next_state, action)]

            self.theta_tilde = np.copy(self._pars_cache["theta"])
            self.omega_tilde = np.copy(self._pars_cache["omega"])
            self.theta = np.copy(self.theta_tilde)
            self.omega = np.copy(self.omega_tilde)

            self.batch_grad_info = copy.deepcopy(self._grad_cache)
            self._grad_cache = {"G": G_x / self.batch_size,
                                "H": H_x / self.batch_size}
            # 这段代码区分每次theta-tilde是取均值还是取上个batch最后一次theta
            self._pars_cache = {"theta": np.copy(self.theta), "omega": np.copy(self.omega)}

        if self._trajectory is None:
            # 前M个样本, 不更新参数
            self._pars_cache["theta"] = np.copy(self.theta)
            self._pars_cache["omega"] = np.copy(self.omega)
            return self.theta, self.omega
        else:
            s, r, s_pine, _a = self._trajectory[np.random.choice(self.batch_size)]
            G_x, H_x = self._extract_grad_info(s, r, s_pine, _a)  # Compute the current gradient info

            alpha = self.eta_theta
            beta = self.eta_omega
            theta = np.copy(self.theta)
            omega = np.copy(self.omega)

            G, H = self._get_batch_grad_info()  # Averaged gradient over this batch
            grad_1_theta = G_x[0]
            grad_1_w = H_x[0]

            G_tmp, H_tmp = self._extract_grad_info_2(self.theta_tilde, self.omega_tilde, s, r, s_pine, _a)
            grad_2_theta, grad_2_w = G_tmp[0], H_tmp[0]

            grad_3_theta = G[0]
            grad_3_w = H[0]

            self.theta = theta + alpha * (grad_1_theta - grad_2_theta + grad_3_theta)
            self.omega = omega + beta * (grad_1_w - grad_2_w + grad_3_w)
            ratio = 1.0
            if np.sum(self.theta ** 2)>= ratio**2:
                self.theta = ratio * self.theta/np.sqrt(np.sum(self.theta ** 2))
            # 这段代码区分每次theta-tilde是取均值还是取上个batch最后一次theta
            # self._pars_cache["theta"] = self._pars_cache["theta"] + self.theta / self.batch_size
            # self._pars_cache["w"] = self._pars_cache["w"] + self.w / self.batch_size
            self._pars_cache["theta"] = np.copy(self.theta)
            self._pars_cache["omega"] = np.copy(self.omega)

        return grad_1_theta - grad_2_theta + grad_3_theta, grad_1_w - grad_2_w + grad_3_w

    def get_grad(self, current_state, reward, next_state, action=None):
        G_x, H_x = self._extract_grad_info_2(self.theta_tilde, self.omega_tilde, current_state, reward, next_state,
                                             action)

        if len(self._trajectory_cache) < self.batch_size:
            # 开始预存这个trajectory里的full gradient info
            return np.zeros_like(self.theta), np.zeros_like(self.omega)
        else:
            # 记录了刚好M个样本的时候. 进入下一个batch
            # cache全部重置, theta-tilde更新成前一个batch的均值, theta重设为theta-tilde
            self._trajectory = copy.deepcopy(self._trajectory_cache)
            self._trajectory_cache = [(current_state, reward, next_state, action)]

            self.theta_tilde = np.copy(self._pars_cache["theta"])
            self.omega_tilde = np.copy(self._pars_cache["omega"])
            self.theta = np.copy(self.theta_tilde)
            self.omega = np.copy(self.omega_tilde)

            self.batch_grad_info = copy.deepcopy(self._grad_cache)
            self._grad_cache = {"G": G_x / self.batch_size,
                                "H": H_x / self.batch_size}
            # 这段代码区分每次theta-tilde是取均值还是取上个batch最后一次theta
            self._pars_cache = {"theta": np.copy(self.theta), "omega": np.copy(self.omega)}

        if self._trajectory is None:
            # 前M个样本, 不更新参数
            self._pars_cache["theta"] = np.copy(self.theta)
            self._pars_cache["omega"] = np.copy(self.omega)
            return self.theta, self.omega
        else:
            s, r, s_pine, _a = self._trajectory[np.random.choice(self.batch_size)]
            G_x, H_x = self._extract_grad_info(s, r, s_pine, _a)  # Compute the current gradient info

            alpha = self.eta_theta
            beta = self.eta_omega
            theta = np.copy(self.theta)
            omega = np.copy(self.omega)

            G, H = self._get_batch_grad_info()  # Averaged gradient over this batch
            grad_1_theta = G_x[0]
            grad_1_w = H_x[0]

            G_tmp, H_tmp = self._extract_grad_info_2(self.theta_tilde, self.omega_tilde, s, r, s_pine, _a)
            grad_2_theta, grad_2_w = G_tmp[0], H_tmp[0]

            grad_3_theta = G[0]
            grad_3_w = H[0]

            # 这段代码区分每次theta-tilde是取均值还是取上个batch最后一次theta
            # self._pars_cache["theta"] = self._pars_cache["theta"] + self.theta / self.batch_size
            # self._pars_cache["w"] = self._pars_cache["w"] + self.w / self.batch_size
            self._pars_cache["theta"] = np.copy(self.theta)
            self._pars_cache["omega"] = np.copy(self.omega)

        return grad_1_theta - grad_2_theta + grad_3_theta, grad_1_w - grad_2_w + grad_3_w

