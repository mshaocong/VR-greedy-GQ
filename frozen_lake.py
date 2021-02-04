import numpy as np
import gym
import utils
from optimizer.tdvanilla import TD
from optimizer.tdc import TDC
from optimizer.vrgreedygq import VRTDC
from optimizer.vrtd import VRTD


np.random.seed(1)
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

num_features = 4
gamma = 0.95
target_policy = utils.get_random_policy(16, 4)
target = np.copy(target_policy)
behavior_policy = utils.get_uniform_policy(16, 4)
feature = utils.get_features(16, num_features)
reward = np.zeros(16)
reward[-1] = 1.0

batch_size = 3000
alpha = 0.1
beta = 0.01

stationary = utils.compute_stationary_dist(state_trans_kernel)
A = np.zeros((num_features, num_features))
C = np.zeros((num_features, num_features))
b = np.zeros((num_features, 1))

for s in range(16):
    As = np.zeros((num_features, num_features))
    Cs = np.zeros((num_features, num_features))
    bs = np.zeros((num_features, 1))
    for s_pine in range(16):
        trans_prob = state_trans_kernel[s, s_pine]
        phi_current_state = feature[s, :].reshape((num_features, 1))
        phi_next_state = feature[s_pine, :].reshape((num_features, 1))
        As += trans_prob * np.matmul(phi_current_state,
                                     np.transpose(gamma * phi_next_state - phi_current_state))
        Cs += trans_prob * np.matmul(phi_current_state, np.transpose(phi_current_state))
        bs += trans_prob * reward[s_pine] * phi_current_state
    A += stationary[s] * As
    b += stationary[s] * bs
    C += stationary[s] * Cs
theta_ast = - np.linalg.inv(A).dot(b)


from garnet import Garnet


class TMP_Env(Garnet):
    # Note: this class is not used to generate trajectory. Simply because the first argument of optimizer class requires
    # an environment containing `behavior_policy` and `features` properties.
    def __init__(self, bp, feat):
        super().__init__(16, 4, 4, 4)
        self.behavior_policy = bp
        self.features = feat


env = gym.make("FrozenLake-v0")
env.reset()
current_state = 0
tmp_env = TMP_Env(behavior_policy, feature)

all_td_hist = []
all_tdc_hist = []
all_vrtdc_hist = []
all_vrtd_hist = []

ini_theta = theta_ast + 0.4*np.random.normal(scale=1.0, size=theta_ast.shape)

td_hist = [np.sum((ini_theta - theta_ast) ** 2)]
tdc_hist = [np.sum((ini_theta - theta_ast) ** 2)]
vrtdc_hist = [np.sum((ini_theta - theta_ast) ** 2)]
vrtd_hist = [np.sum((ini_theta - theta_ast) ** 2)]

import time
for _ in range(100):
    td = TD(tmp_env, alpha=alpha, target_policy=target, gamma=gamma)
    td.set_theta(ini_theta)

    tdc = TDC(tmp_env, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    tdc.set_theta(ini_theta)

    vrtdc = VRTDC(tmp_env, batch_size=batch_size, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc.set_theta(ini_theta)

    vrtd = VRTD(tmp_env, batch_size=batch_size, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd.set_theta(ini_theta)

    td_hist = [np.sum((td.theta - theta_ast) ** 2)]
    tdc_hist = [np.sum((td.theta - theta_ast) ** 2)]
    vrtdc_hist = [np.sum((td.theta - theta_ast) ** 2)]
    vrtd_hist = [np.sum((td.theta - theta_ast) ** 2)]
    count = 1

    print("Start Training. Simulation:", _ + 1)
    train_start = time.time()
    for i in range(50000):
        random_action = env.action_space.sample()
        new_state, reward, done, info = env.step(random_action)

        next_state = new_state
        action = random_action

        if count <= batch_size:
            vrtdc_hist.append(np.sum((vrtdc.theta - theta_ast) ** 2))
        else:
            if count % batch_size == 0:
                vrtdc_hist = vrtdc_hist + [np.sum((vrtdc.theta - theta_ast) ** 2) for _ in range(batch_size)]
            elif count % batch_size != 0:
                vrtdc_hist.append(np.sum((vrtdc.theta - theta_ast) ** 2))
                vrtdc_hist.append(np.sum((vrtdc.theta - theta_ast) ** 2))
        vrtdc.update(current_state, reward, next_state, action)

        if count <= batch_size:
            vrtd_hist.append(np.sum((vrtd.theta - theta_ast) ** 2))
        else:
            if count % batch_size == 0:
                vrtd_hist = vrtd_hist + [np.sum((vrtd.theta - theta_ast) ** 2) for _ in range(batch_size)]
            elif count % batch_size != 0:
                vrtd_hist.append(np.sum((vrtd.theta - theta_ast) ** 2))
                vrtd_hist.append(np.sum((vrtd.theta - theta_ast) ** 2))
        vrtd.update(current_state, reward, next_state, action)

        tdc.update(current_state, reward, next_state, action)
        tdc_hist.append(np.sum((tdc.theta - theta_ast) ** 2))

        td.update(current_state, reward, next_state, action)
        td_hist.append(np.sum((td.theta - theta_ast) ** 2))

        current_state = new_state
        count += 1
        if (i + 1) % 10000 == 0:
            print("Current iteration:", i + 1, ". Time Spent:", time.time() - train_start)
            train_start = time.time()

        if done:
            env.reset()
            current_state = 0
    all_td_hist.append(td_hist)
    all_tdc_hist.append(tdc_hist)
    all_vrtdc_hist.append(vrtdc_hist)
    all_vrtd_hist.append(vrtd_hist)

import matplotlib.pyplot as plt
hist_td = all_td_hist
hist_tdc = all_tdc_hist
hist_vrtdc = all_vrtdc_hist
hist_vrtd = all_vrtd_hist
plt.figure()
utils.easy_plot(hist_tdc, "orange", "TDC")
utils.easy_plot(hist_td, "g", "TD")
utils.easy_plot(hist_vrtd, "b", "VRTD: M=3000", cut_off=len(hist_td[0]))
utils.easy_plot(hist_vrtdc, "r", "VRTDC: M=3000", cut_off=len(hist_td[0]))

plt.legend(loc=1)
plt.ylabel(r"$||\theta - \theta^\ast ||^2$")
plt.xlabel("# of gradient computations")
#plt.savefig('fig1_frozen_lake.png', dpi=300)
plt.show()
#import pickle
#with open('hist_frozen_lake.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([hist_td, hist_tdc, hist_vrtdc, hist_vrtd], f)