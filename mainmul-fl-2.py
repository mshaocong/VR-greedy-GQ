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

def _simulation(tmp_env, ini_theta, alpha, beta, batch_size, trajectory_length=50000, gamma=0.95, target=None):
    env = gym.make("FrozenLake-v0")
    env.reset()
    current_state = 0

    gq = GreedyGQ_Base(tmp_env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    gq.set_theta(ini_theta)
    for i in range(trajectory_length):
        random_action = env.action_space.sample()
        new_state, reward, done, info = env.step(random_action)
        next_state = new_state
        action = random_action

        gq.update(current_state, reward, next_state, action)

        if done:
            env.reset()
            current_state = 0
    return gq.theta[0]
    # return gq_hist_last, vrgq_hist_last

def easy_simulation(env1, alpha, beta, batch_size, trajectory_length=50000, num_simulation=100, gamma=0.95, target=None):
    env = gym.make("FrozenLake-v0")
    env.reset()
    ini_theta = np.random.normal(scale=2.0, size=env1.num_features )

    params = (env1, ini_theta, alpha, beta, batch_size, trajectory_length, gamma, target)

    print("Start Training.")
    train_start = time.time()
    pool = Pool(3)
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
    max_num_iteration = 100000
    batch_size = 1000
    alpha = 0.02
    beta = 0.01
    num_simulation = 3 #60

    out = easy_simulation(tmp_env, alpha, beta, batch_size, trajectory_length=max_num_iteration,
                          num_simulation=num_simulation, gamma=gamma, target=None)

    print( evaluate_J(tmp_env, out[0]))
    print( evaluate_J(tmp_env, out[1]))
    print( evaluate_J(tmp_env, out[2]))
    return out


if __name__ == "__main__":
    out = main()
    print(out)
