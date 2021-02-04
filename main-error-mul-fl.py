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

    vrgq = VRGreedyGQ(tmp_env, batch_size=batch_size, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    vrgq.set_theta(ini_theta)

    vrgq_hist_avg = [evaluate_J(tmp_env, vrgq.theta)]
    for i in range(trajectory_length):
        random_action = env.action_space.sample()
        new_state, reward, done, info = env.step(random_action)
        next_state = new_state
        action = random_action

        vrgq.update(current_state, reward, next_state, action)
        if i>batch_size:
            if i % 10000 == 0:
                print(i)
                vrgq_hist_avg.append(np.min(vrgq_hist_avg + [evaluate_J(tmp_env, vrgq.theta)]))

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
    pool = Pool(3)
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

    #state_trans_kernel = np.zeros((16, 16))
    #state_trans_kernel[0, (1, 4)] = 0.25
    #state_trans_kernel[0, 0] = 0.5
    #state_trans_kernel[1, (0, 1, 2, 5)] = 0.25
    #state_trans_kernel[2, (1, 2, 3, 6)] = 0.25
    #state_trans_kernel[3, (2, 7)] = 0.25
    #state_trans_kernel[3, 3] = 0.5
    #state_trans_kernel[4, (0, 4, 5, 8)] = 0.25
    #state_trans_kernel[5, (1,4,6,9)] = 0.25
    #state_trans_kernel[6, (2, 5, 7, 10)] = 0.25
    #state_trans_kernel[7, (3, 6, 11, 7)] = 0.25
    #state_trans_kernel[8, (4, 8, 9, 12)] = 0.25
    #state_trans_kernel[9, (5, 8, 11, 13)] = 0.25
    #state_trans_kernel[10, (6, 9, 11, 14)] = 0.25
    #state_trans_kernel[11, (7, 10, 11, 15)] = 0.25
    #state_trans_kernel[12, (8, 13)] = 0.5
    #state_trans_kernel[13, (9, 12, 13, 14)] = 0.25
    #state_trans_kernel[14, (10, 13, 14, 15)] = 0.25
    #state_trans_kernel[15, :] = 1.0/16.0

    num_features = 8
    behavior_policy = utils.get_uniform_policy(16, 4)
    feature = utils.get_features(4,16, num_features)
    reward = np.zeros(16)
    reward[-1] = 1.0
    #reward[5] = -0.05
    #reward[7] = -0.05
    #reward[11] = -0.05
    #reward[12] = -0.05
    #reward[0] = 0.03
    #reward[1] = 0.03
    #reward[2] = 0.03
    #reward[3] = 0.03
    #reward[4] = 0.03
    #reward[6] = 0.03
    #reward[8] = 0.03
    #reward[9] = 0.03
    #reward[10] = 0.03
    #reward[13] = 0.03
    #reward[14] = 0.03

    tmp_env = TMP_Env(behavior_policy, feature)

    gamma = 0.95
    max_num_iteration = 200000
    batch_size = 3000
    alpha = 0.2
    beta = 0.1
    num_simulation = 1 #60
    for batch_size in [100, 300]:
        out = main(tmp_env, alpha, beta, batch_size, max_num_iteration, num_simulation, gamma=0.95, target=None)

        with open('hist-error-'+str(batch_size)+'-fl.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(out, f)

