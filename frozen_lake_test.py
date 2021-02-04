import numpy as np
import gym
import utils
from optimizer.tdvanilla import TD
from optimizer.tdc import TDC
from optimizer.vrgreedygq import VRTDC
from optimizer.vrtd import VRTD
import time
import matplotlib.pyplot as plt

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

ini_theta = theta_ast + 0.6 * np.random.normal(scale=1.0, size=theta_ast.shape)
all_vrtdc1_hist = []
all_vrtd1_hist = []
all_vrtdc1000_hist = []
all_vrtd1000_hist = []
all_vrtdc2000_hist = []
all_vrtd2000_hist = []
all_vrtdc3000_hist = []
all_vrtd3000_hist = []
all_vrtdc4000_hist = []
all_vrtd4000_hist = []
all_vrtdc5000_hist = []
all_vrtd5000_hist = []

num_simulation = 100
for _ in range(num_simulation):
    env.reset()
    current_state = 0

    vrtdc1 = TDC(tmp_env, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc1.set_theta(ini_theta)
    vrtd1 = TD(tmp_env, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd1.set_theta(ini_theta)

    vrtdc1000 = VRTDC(tmp_env, batch_size=1000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc1000.set_theta(ini_theta)
    vrtd1000 = VRTD(tmp_env, batch_size=1000, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd1000.set_theta(ini_theta)

    vrtdc2000 = VRTDC(tmp_env, batch_size=2000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc2000.set_theta(ini_theta)
    vrtd2000 = VRTD(tmp_env, batch_size=2000, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd2000.set_theta(ini_theta)

    vrtdc3000 = VRTDC(tmp_env, batch_size=3000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc3000.set_theta(ini_theta)
    vrtd3000 = VRTD(tmp_env, batch_size=3000, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd3000.set_theta(ini_theta)

    vrtdc4000 = VRTDC(tmp_env, batch_size=4000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc4000.set_theta(ini_theta)
    vrtd4000 = VRTD(tmp_env, batch_size=4000, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd4000.set_theta(ini_theta)

    vrtdc5000 = VRTDC(tmp_env, batch_size=5000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc5000.set_theta(ini_theta)
    vrtd5000 = VRTD(tmp_env, batch_size=5000, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd5000.set_theta(ini_theta)

    print("Start Training. Simulation:", _ + 1)
    train_start = time.time()

    vrtdc1_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtdc1000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtdc2000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtdc3000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtdc4000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtdc5000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]

    vrtd1_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtd1000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtd2000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtd3000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtd4000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    vrtd5000_hist = [np.sum((vrtdc1.theta - theta_ast) ** 2)]
    count = 1
    for i in range(50000):
        random_action = env.action_space.sample()
        next_state, reward, done, info = env.step(random_action)
        action = random_action


        vrtdc1.update(current_state, reward, next_state, action)
        vrtdc1_hist.append(np.sum((vrtdc1.theta - theta_ast) ** 2))
        vrtd1.update(current_state, reward, next_state, action)
        vrtd1_hist.append(np.sum((vrtd1.theta - theta_ast) ** 2))

        vrtdc1000.update(current_state, reward, next_state, action)
        vrtdc1000_hist.append(np.sum((vrtdc1000.theta - theta_ast) ** 2))
        vrtd1000.update(current_state, reward, next_state, action)
        vrtd1000_hist.append(np.sum((vrtd1000.theta - theta_ast) ** 2))

        vrtdc2000.update(current_state, reward, next_state, action)
        vrtdc2000_hist.append(np.sum((vrtdc2000.theta - theta_ast) ** 2))
        vrtd2000.update(current_state, reward, next_state, action)
        vrtd2000_hist.append(np.sum((vrtd2000.theta - theta_ast) ** 2))

        vrtdc3000.update(current_state, reward, next_state, action)
        vrtdc3000_hist.append(np.sum((vrtdc3000.theta - theta_ast) ** 2))
        vrtd3000.update(current_state, reward, next_state, action)
        vrtd3000_hist.append(np.sum((vrtd3000.theta - theta_ast) ** 2))

        vrtdc4000.update(current_state, reward, next_state, action)
        vrtdc4000_hist.append(np.sum((vrtdc4000.theta - theta_ast) ** 2))
        vrtd4000.update(current_state, reward, next_state, action)
        vrtd4000_hist.append(np.sum((vrtd4000.theta - theta_ast) ** 2))

        vrtdc5000.update(current_state, reward, next_state, action)
        vrtdc5000_hist.append(np.sum((vrtdc5000.theta - theta_ast) ** 2))
        vrtd5000.update(current_state, reward, next_state, action)
        vrtd5000_hist.append(np.sum((vrtd5000.theta - theta_ast) ** 2))

        current_state = np.copy(next_state)
        if (i + 1) % 10000 == 0:
            print("Current iteration:", i + 1, ". Time Spent:", time.time() - train_start)
            train_start = time.time()
        count += 1

        if done:
            env.reset()
            current_state = 0
    all_vrtdc1_hist.append(vrtdc1_hist)
    all_vrtdc1000_hist.append(vrtdc1000_hist)
    all_vrtdc2000_hist.append(vrtdc2000_hist)
    all_vrtdc3000_hist.append(vrtdc3000_hist)
    all_vrtdc4000_hist.append(vrtdc4000_hist)
    all_vrtdc5000_hist.append(vrtdc5000_hist)
    all_vrtd1_hist.append(vrtd1_hist)
    all_vrtd1000_hist.append(vrtd1000_hist)
    all_vrtd2000_hist.append(vrtd2000_hist)
    all_vrtd3000_hist.append(vrtd3000_hist)
    all_vrtd4000_hist.append(vrtd4000_hist)
    all_vrtd5000_hist.append(vrtd5000_hist)

h1, h2, h3, h4, h5, h6, tdh1, tdh2, tdh3, tdh4, tdh5, tdh6 = (
    all_vrtdc1_hist, all_vrtdc1000_hist, all_vrtdc2000_hist, all_vrtdc3000_hist, all_vrtdc4000_hist, all_vrtdc5000_hist,
    all_vrtd1_hist, all_vrtd1000_hist, all_vrtd2000_hist, all_vrtd3000_hist, all_vrtd4000_hist, all_vrtd5000_hist)

fig, ax = plt.subplots()
vrtdc_h = [h1, h2, h3, h4, h5, h6]
vrtd_h = [tdh1, tdh2, tdh3, tdh4, tdh5, tdh6]


num_obs = 10000
errors_vrtdc = [np.mean(np.array(h)[:, -num_obs:], axis=1) for h in vrtdc_h]
errors_vrtd = [np.mean(np.array(h)[:, -num_obs:], axis=1) for h in vrtd_h]

import pandas as pd

batch_size_list = ['1', "1000", "2000", "3000", "4000", "5000"]
DF = pd.DataFrame(columns=["Errors", "Batch Size", "Algorithm"])
count = 0
for i in range(len(batch_size_list)):
    for j in range(num_simulation):
        DF.loc[count] = [errors_vrtdc[i][j], batch_size_list[i], "VRTDC"]
        count += 1
        DF.loc[count] = [errors_vrtd[i][j], batch_size_list[i], "VRTD"]
        count += 1

import seaborn as sns

sns.set(style="ticks", palette="pastel")
bp = sns.boxplot(x="Batch Size", y="Errors", data=DF, hue="Algorithm", palette=["red", "blue"],
                 order=batch_size_list, showmeans=True)

fig = bp.get_figure()
fig.savefig("fig2_frozen_lake3.png", dpi=300)
DF.to_csv("out_frozen_lake3.csv")