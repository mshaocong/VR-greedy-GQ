import gym
import matplotlib.pyplot as plt
from garnet import *
from utils import *
from optimizer.tdvanilla import TD
from optimizer.tdc import TDC
from optimizer.vrgreedygq import VRTDC
from optimizer.vrtd import VRTD
import time


def easy_simulation(env, alpha, beta, trajectory_length=50000, num_simulation=100, gamma=0.95,
                    target=None):
    ini_start = time.time()

    print("Initialization...")
    A, b, C = evaluate_AbC(env, gamma=gamma, target_policy=target)
    theta_ast = -np.matmul(np.linalg.inv(A), b)

    ini_theta = theta_ast + 0.8 * np.random.normal(scale=1.0, size=theta_ast.shape)
    print("Optimal Theta:", theta_ast)
    print("Initial Theta:", ini_theta)
    print("Initialization Completed. Time Spent:", time.time() - ini_start)

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
    for _ in range(num_simulation):
        env.reset()
        current_state = env.current_state

        vrtdc1 = TDC(env, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        vrtdc1.set_theta(ini_theta)
        vrtd1 = TD(env, alpha=alpha, target_policy=target, gamma=gamma)
        vrtd1.set_theta(ini_theta)

        vrtdc1000 = VRTDC(env, batch_size=1000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        vrtdc1000.set_theta(ini_theta)
        vrtd1000 = VRTD(env, batch_size=1000, alpha=alpha, target_policy=target, gamma=gamma)
        vrtd1000.set_theta(ini_theta)

        vrtdc2000 = VRTDC(env, batch_size=2000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        vrtdc2000.set_theta(ini_theta)
        vrtd2000 = VRTD(env, batch_size=2000, alpha=alpha, target_policy=target, gamma=gamma)
        vrtd2000.set_theta(ini_theta)

        vrtdc3000 = VRTDC(env, batch_size=3000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        vrtdc3000.set_theta(ini_theta)
        vrtd3000 = VRTD(env, batch_size=3000, alpha=alpha, target_policy=target, gamma=gamma)
        vrtd3000.set_theta(ini_theta)

        vrtdc4000 = VRTDC(env, batch_size=4000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        vrtdc4000.set_theta(ini_theta)
        vrtd4000 = VRTD(env, batch_size=4000, alpha=alpha, target_policy=target, gamma=gamma)
        vrtd4000.set_theta(ini_theta)

        vrtdc5000 = VRTDC(env, batch_size=5000, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        vrtdc5000.set_theta(ini_theta)
        vrtd5000 = VRTD(env, batch_size=5000, alpha=alpha, target_policy=target, gamma=gamma)
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
        for i in range(trajectory_length):
            next_state, reward, action = env.step()
            # action = None

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
    return all_vrtdc1_hist, all_vrtdc1000_hist, all_vrtdc2000_hist, all_vrtdc3000_hist, all_vrtdc4000_hist, all_vrtdc5000_hist, \
           all_vrtd1_hist, all_vrtd1000_hist, all_vrtd2000_hist, all_vrtd3000_hist, all_vrtd4000_hist, all_vrtd5000_hist


np.random.seed(91)

# Compare Different Batch Sizes
num_states = 500
num_actions = 20
branching_factor = 50
num_features = 15

print("Set Up the Simulation Environment...")
env = Garnet(num_states, num_actions, branching_factor, num_features)
print("Done.")

gamma = 0.95
max_num_iteration = 100000
alpha = 0.1
beta = 0.02
target = get_random_policy(num_states, num_actions)
num_simulation = 250

h1, h2, h3, h4, h5, h6, tdh1, tdh2, tdh3, tdh4, tdh5, tdh6 = easy_simulation(env, alpha, beta,
                                                                             trajectory_length=max_num_iteration,
                                                                             num_simulation=num_simulation, gamma=gamma,
                                                                             target=target)

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
bp = sns.boxplot(x="Batch Size", y="Errors", data=DF, hue="Algorithm", palette=["red", "blue"])
fig = bp.get_figure()
fig.savefig("fig2-corrected.png", dpi=300)

DF.to_csv("out-corrected.csv")