import gym
import matplotlib.pyplot as plt
from garnet import *
from utils import *
from optimizer.greedygq import GreedyGQ_Base
from optimizer.vrgreedygq import VRGreedyGQ
import time
from multiprocessing import Pool
import itertools

import pickle

class TMP_Env(Garnet):
    # Note: this class is not used to generate trajectory. Simply because the first argument of optimizer class requires
    # an environment containing `behavior_policy` and `features` properties.
    def __init__(self, bp, feat):
        super().__init__(16, 4, 4, 8)
        self.behavior_policy = bp
        self.features = feat

def _simulation(env, ini_theta, alpha, beta, batch_size, trajectory_length=50000, gamma=0.95, target=None):
    train_start = time.time()
    env.reset()
    current_state = env.current_state

    gq = GreedyGQ_Base(env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    gq.set_theta(ini_theta)

    vrgq = VRGreedyGQ(env, batch_size=batch_size, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    vrgq.set_theta(ini_theta)

    gq_var = []
    vrgq_var = []
    count = 1
    num_sample_mc = 50
    for i in range(trajectory_length):
        next_state, reward, action = env.step()

        gq.update(current_state, reward, next_state, action)
        vrgq.update(current_state, reward, next_state, action)

        if i >= batch_size:
            if i % 1000 == 0:
                estimated_var = 0.0
                tmp_theta = np.copy(vrgq.theta)
                tmp_w = np.copy(vrgq.omega)
                true_grad = evaluate_J(env, tmp_theta)
                for ddd in range(num_sample_mc):
                    ss, aa, next_ss, rr = env.sample()
                    grad_theta, grad_omega = vrgq.get_grad(ss, rr, next_ss, aa)
                    estimated_var += np.sum((grad_theta - true_grad) ** 2) / num_sample_mc
                vrgq_var.append(estimated_var)

        if i % 1000 == 0:
            estimated_var = 0.0
            tmp_theta = np.copy(gq.theta[0])
            tmp_w = np.copy(gq.omega)
            true_grad = evaluate_J(env, tmp_theta)
            for ddd in range(num_sample_mc):
                ss, aa, next_ss, rr = env.sample()
                grad_theta, grad_omega = gq._extract_grad_info(ss, rr, next_ss, aa)
                estimated_var += np.sum((grad_theta - true_grad) ** 2) / num_sample_mc
            gq_var.append(estimated_var)

        if (i + 1) % 10000 == 0:
            print("Current iteration:", i + 1, ". Time Spent:", time.time() - train_start)
            train_start = time.time()
        count += 1

        current_state = np.copy(next_state)
    return gq_var, vrgq_var
    # return gq_hist_last, vrgq_hist_last

def easy_simulation(env, alpha, beta, batch_size, trajectory_length=50000, num_simulation=100, gamma=0.95, target=None):
    ini_start = time.time()

    print("Initialization...")
    ini_theta = np.random.normal(scale=2.0, size=env.num_features )
    #_gq = GreedyGQ_Base(env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    #_gq.set_theta(ini_theta)
    #env.reset()
    #current_state = env.current_state
    #for _ in range(10):
    #    next_state, reward, action = env.step()
    #    _gq.update(current_state, reward, next_state, action)
    #    current_state = np.copy(next_state)
    #ini_theta = _gq.theta
    print("Initialization Completed. Time Spent:", time.time() - ini_start)

    params = (env, ini_theta, alpha, beta, batch_size, trajectory_length, gamma, target)

    print("Start Training.")
    train_start = time.time()
    pool = Pool(6)
    out = pool.starmap(_simulation, itertools.repeat(params, num_simulation))
    pool.close()
    pool.join()
    print("Training complete. Time Spent:", time.time() - train_start)
    return out


def main():
    np.random.seed(114514)

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

    num_features = 8
    behavior_policy = utils.get_uniform_policy(16, 4)
    feature = utils.get_features(4,16, num_features)
    reward = np.zeros(16)
    reward[-1] = 1.0

    tmp_env = TMP_Env(behavior_policy, feature)

    gamma = 0.95
    max_num_iteration = 200000
    batch_size = 3000
    alpha = 0.2
    beta = 0.1
    num_simulation = 3 #60

    out = easy_simulation(tmp_env, alpha, beta, batch_size, trajectory_length=max_num_iteration,
                          num_simulation=num_simulation, gamma=gamma, target=None)

    return out


if __name__ == "__main__":
    out = main()

    hist_gq = []
    hist_vrgq = []
    for output in out:
        hist_gq.append(output[0])
        hist_vrgq.append(output[1])

    with open('hist-var-60.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([hist_gq, hist_vrgq], f)

    plt.plot(np.average(hist_gq, axis=0),c="r")
    plt.plot(np.average(hist_vrgq, axis=0),c="b")
    plt.ylim(0,1)
    plt.show()

