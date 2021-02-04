import gym
import matplotlib.pyplot as plt
from garnet import *
from utils import *
from optimizer.tdvanilla import TD
from optimizer.tdc import TDC
from optimizer.vrgreedygq import VRTDC
from optimizer.vrtd import VRTD
import time
from multiprocessing import Pool
import itertools
import pandas as pd
import seaborn as sns


def _simulation(env, ini_theta, theta_ast, bs_list=None, alpha=0.1, beta=0.02,
                trajectory_length=50000, gamma=0.95, target=None):
    if bs_list is None:
        bs_list = [1000, 2000, 3000, 4000, 5000]
    env.reset()
    current_state = env.current_state

    vrtdc1 = TDC(env, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
    vrtdc1.set_theta(ini_theta)

    vrtd1 = TD(env, alpha=alpha, target_policy=target, gamma=gamma)
    vrtd1.set_theta(ini_theta)

    vrtdc_opt_dict = {1: vrtdc1}
    vrtd_opt_dict = {1: vrtd1}

    vrtd_hist_dict = {k: [np.sum((ini_theta - theta_ast) ** 2)] for k in [1] + bs_list}
    vrtdc_hist_dict = {k: [np.sum((ini_theta - theta_ast) ** 2)] for k in [1] + bs_list}
    for bs in bs_list:
        opt_vrtdc = VRTDC(env, batch_size=bs, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        opt_vrtdc.set_theta(ini_theta)
        vrtdc_opt_dict[bs] = opt_vrtdc

        opt_vrtd = VRTD(env, batch_size=bs, alpha=alpha, target_policy=target, gamma=gamma)
        opt_vrtd.set_theta(ini_theta)
        vrtd_opt_dict[bs] = opt_vrtd

    count = 1
    for i in range(trajectory_length):
        next_state, reward, action = env.step()
        for bs in [1] + bs_list:
            vrtd_opt_dict[bs].update(current_state, reward, next_state, action)
            vrtdc_opt_dict[bs].update(current_state, reward, next_state, action)

            vrtd_hist_dict[bs].append(np.sum((vrtd_opt_dict[bs].theta - theta_ast) ** 2))
            vrtdc_hist_dict[bs].append(np.sum((vrtdc_opt_dict[bs].theta - theta_ast) ** 2))

        current_state = np.copy(next_state)
        count += 1
    return vrtd_hist_dict, vrtdc_hist_dict


def easy_simulation(env, alpha, beta, bs_list=None, trajectory_length=50000, num_simulation=100, gamma=0.95,
                    target=None):
    ini_start = time.time()
    if bs_list is None:
        bs_list = [1000, 2000, 3000, 4000, 5000]

    print("Initialization...")
    A, b, C = evaluate_AbC(env, gamma=gamma, target_policy=target)
    theta_ast = -np.matmul(np.linalg.inv(A), b)

    theta_ast + 0.8 * np.random.normal(scale=1.0, size=theta_ast.shape)
    print("Optimal Theta:", theta_ast)
    print("Initial Theta:", ini_theta)
    print("Initialization Completed. Time Spent:", time.time() - ini_start)

    params = (env, ini_theta, theta_ast, bs_list, alpha, beta, trajectory_length, gamma, target)

    print("Start Training.")
    train_start = time.time()
    pool = Pool()
    out = pool.starmap(_simulation, itertools.repeat(params, num_simulation))
    pool.close()
    pool.join()
    print("Training complete. Time Spent:", time.time() - train_start)
    return out


def main():
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
    num_simulation = 100

    out = easy_simulation(env, alpha, beta,
                          trajectory_length=max_num_iteration,
                          num_simulation=num_simulation, gamma=gamma,
                          target=target)

    return out


if __name__ == "__main__":
    out = main()
    num_obs = 10000
    DF = pd.DataFrame(columns=["Errors", "Batch Size", "Algorithm"])
    count = 0
    for output in out:
        vrtd_hist = output[0]
        vrtdc_hist = output[1]
        for bs in [1, 1000, 2000, 3000, 4000, 5000]:
            DF.loc[count] = [np.mean(np.array(vrtdc_hist[bs])[-num_obs:]), str(bs), "VRTDC"]
            count += 1
            DF.loc[count] = [np.mean(np.array(vrtd_hist[bs])[-num_obs:]), str(bs), "VRTD"]
            count += 1

    sns.set(style="ticks", palette="pastel")
    bp = sns.boxplot(x="Batch Size", y="Errors", data=DF, hue="Algorithm", palette=["red", "blue"])
    fig = bp.get_figure()
    fig.savefig("fig2-corrected-mul.png", dpi=300)
    DF.to_csv("out-corrected-mul.csv")

