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

def _simulation(env, ini_theta, alpha, beta, batch_size, trajectory_length=50000, gamma=0.95, target=None):
    env.reset()
    current_state = env.current_state

    vrgq = VRGreedyGQ(env, batch_size=batch_size, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    vrgq.set_theta(ini_theta)

    vrgq_hist_avg = [evaluate_J(env, vrgq.theta)]
    for i in range(trajectory_length):
        next_state, reward, action = env.step()

        vrgq.update(current_state, reward, next_state, action)
        if i>batch_size:
            vrgq_hist_avg.append((len(vrgq_hist_avg)  * vrgq_hist_avg[-1] + evaluate_J(env, vrgq.theta)) /(len(vrgq_hist_avg) + 1))


        current_state = np.copy(next_state)
    return vrgq_hist_avg
    # return gq_hist_last, vrgq_hist_last

def easy_simulation(env, alpha, beta, batch_size, trajectory_length=50000, num_simulation=100, gamma=0.95, target=None):
    ini_start = time.time()

    print("Initialization...")
    ini_theta = np.random.normal(scale=1.0, size=env.num_features )
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


def main(env, alpha, beta, batch_size, trajectory_length,
                          num_simulation, gamma, target):
    np.random.seed(114514)

    out = easy_simulation(env, alpha, beta, batch_size, trajectory_length,
                          num_simulation=num_simulation, gamma=gamma, target=target)

    return out


if __name__ == "__main__":
    num_states = 5
    num_actions = 3
    branching_factor = 2
    num_features = 3

    print("Set Up the Simulation Environment...")
    env = Garnet(num_states, num_actions, branching_factor, num_features)
    print("Done.")

    trajectory_length = 100000
    alpha = 0.02
    beta = 0.01
    num_simulation = 60 #16


    for batch_size in [1000, 2000, 3000]:
        env.reset()

        out = main(env, alpha, beta, batch_size, trajectory_length, num_simulation, gamma=0.95, target=None)

        with open('hist-error-'+str(batch_size)+'-2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(out, f)

