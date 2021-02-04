import gym
import matplotlib.pyplot as plt
from garnet import *
from utils import *
from optimizer.greedygq import GreedyGQ_Base
from optimizer.vrgreedygq import VRGreedyGQ
import time
from multiprocessing import Pool
import itertools


def _easy_simulation(env, alpha, beta, batch_size, trajectory_length=50000, num_simulation=100, gamma=0.95,
                    target=None):
    ini_start = time.time()

    print("Initialization...")
    ini_theta = np.random.normal(scale=3.0, size=env.num_features )
    print("Initialization Completed. Time Spent:", time.time() - ini_start)
    stationary = compute_stationary_dist(env.trans_kernel)

    all_gq_hist_min = []
    all_vrgq_hist_min = []

    all_gq_hist_last = []
    all_vrgq_hist_last = []

    all_gq_hist_avg = []
    all_vrgq_hist_avg = []

    for _ in range(num_simulation):
        env.reset()
        current_state = env.current_state

        estimate1 = GreedyGQ_Base(env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
        estimate1.set_theta(ini_theta)

        estimate2 = GreedyGQ_Base(env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
        estimate2.set_theta(ini_theta)

        gq = GreedyGQ_Base(env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
        gq.set_theta(ini_theta)

        vrgq = VRGreedyGQ(env, batch_size=batch_size, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
        vrgq.set_theta(ini_theta)

        print("Start Training. Simulation:", _ + 1)
        train_start = time.time()

        gq_hist_last = [evaluate_J(env, gq.theta)]
        vrgq_hist_last = [evaluate_J(env, gq.theta)]
        gq_hist_min = [evaluate_J(env, gq.theta)]
        vrgq_hist_min = [evaluate_J(env, gq.theta)]
        count = 1
        for i in range(trajectory_length):
            next_state, reward, action = env.step()

            grad_theta_gq, grad_omega_gq = gq.update(current_state, reward, next_state, action)
            gq_hist_min.append(np.min(gq_hist_min + [evaluate_J(env, gq.theta[0])]))
            gq_hist_last.append(evaluate_J(env, gq.theta[0]) )

            grad_theta_vrgq, grad_omega_vrgq = vrgq.update(current_state, reward, next_state, action)
            vrgq_hist_min.append( np.min(vrgq_hist_min + [evaluate_J(env, vrgq.theta)]))
            vrgq_hist_last.append( evaluate_J(env, vrgq.theta))

            # Estimate the variance
            grad_theta = np.zeros_like(grad_theta_gq)
            grad_omega = np.zeros_like(grad_omega_gq)
            estimate1.set_theta(gq.theta)
            estimate1.set_omega(gq.omega)
            estimate2.set_theta(vrgq.theta)
            estimate2.set_omega(vrgq.omega)
            for sss in env.state_space:
                pass

            current_state = np.copy(next_state)
            if (i + 1) % 10000 == 0:
                print("Current iteration:", i + 1, ". Time Spent:", time.time() - train_start)
                train_start = time.time()
            count += 1
        all_gq_hist_min.append(gq_hist_min)
        all_vrgq_hist_min.append(vrgq_hist_min)
        all_gq_hist_last.append(gq_hist_last)
        all_vrgq_hist_last.append(vrgq_hist_last)
    return all_gq_hist_last, all_vrgq_hist_last

def easy_simulation(env, alpha, beta, batch_size, trajectory_length=50000, num_simulation=100, gamma=0.95,
                    target=None):
    ini_start = time.time()

    print("Initialization...")
    ini_theta = np.random.normal(scale=3.0, size=env.num_features )
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
    np.random.seed(114514)

    # Compare TD, TDC, and VRTDC
    num_states = 5
    num_actions = 3
    branching_factor = 2
    num_features = 3

    print("Set Up the Simulation Environment...")
    env = Garnet(num_states, num_actions, branching_factor, num_features)
    print("Done.")

gamma = 0.95
max_num_iteration = 20000
batch_size = 2000
alpha = 0.2
beta = 0.02
target = get_random_policy(num_states, num_actions)
num_simulation = 10


hist_gq, hist_vrgq = easy_simulation(env, alpha, beta, batch_size, trajectory_length=max_num_iteration,
                                                        num_simulation=num_simulation, gamma=gamma, target=target)


plt.plot(np.average(hist_gq, axis=0))
plt.plot(np.average(hist_vrgq, axis=0))
plt.ylim(0,1)
plt.show()

raise

"""import pickle
with open('hist.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([hist_td, hist_tdc, hist_vrtdc, hist_vrtd], f)
"""
plt.figure()
easy_plot(hist_tdc, "orange", "TDC")
easy_plot(hist_td, "g", "TD")
easy_plot(hist_vrtd, "b", "VRTD: M=3000", cut_off=len(hist_td[0]))
easy_plot(hist_vrtdc, "r", "VRTDC: M=3000", cut_off=len(hist_td[0]))

plt.legend(loc=1)
plt.ylabel(r"$||\theta - \theta^\ast ||^2$")
plt.xlabel("# of gradient computations")
# plt.savefig('fig1.png', dpi=300)
plt.show()
