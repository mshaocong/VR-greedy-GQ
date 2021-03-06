import gym
import matplotlib.pyplot as plt
from garnet import *
from environment import FrozenLake
from utils import *
from optimizer.greedygq import GreedyGQ_Base
from optimizer.vrgreedygq import VRGreedyGQ
from optimizer.pg import PolicyGradient
from optimizer.ac import ActorCritic
import time
from multiprocessing import Pool
import itertools

import pickle

def _simulation(env, ini_theta, alpha, beta, batch_size, trajectory_length=50000, gamma=0.95, target=None):
    train_start = time.time()
    env.reset()
    current_state = env.current_state

    pg = ActorCritic(env, eta_theta=alpha, eta_omega=beta, gamma=gamma, is_on_policy = False)
    pg.set_theta(ini_theta)


    gq = GreedyGQ_Base(env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    gq.set_theta(ini_theta)

    vrgq = VRGreedyGQ(env, batch_size=batch_size, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    vrgq.set_theta(ini_theta)

    env2 = gym.make("FrozenLake-v0", is_slippery=False)
    env2.reset()
    current_state2 = 0
    pppp = SoftmaxPolicy(gq.theta, env, 1.0)
    N = 1000
    r = 0
    for iii in range(N):
        random_action = pppp.get_action(current_state2)
        new_state, reward, done, info = env2.step(random_action)
        if done:
            env2.reset()
            current_state2 = 0
        else:
            current_state2 = new_state
        r += reward
    r /= 1000.0

    r_gq = [np.copy(r)]
    r_vrgq = [np.copy(r)]
    r_pg = [np.copy(r)]

    for i in range(trajectory_length):
        next_state, reward, action = env.step()

        if i % 1000 == 0:
            pg.update(current_state, action)
            env2 = gym.make("FrozenLake-v0", is_slippery=False)
            env2.reset()
            current_state2 = 0
            pppp = SoftmaxPolicy(pg.theta, env, 1.0)
            N = 1000
            r = 0
            for iii in range(N):
                random_action = pppp.get_action(current_state2)
                new_state, reward, done, info = env2.step(random_action)
                if done:
                    env2.reset()
                    current_state2 = 0
                else:
                    current_state2 = new_state
                r += reward
            r /= 1000.0
            r_pg.append(np.max(r_pg + [np.copy(r)]))

        gq.update(current_state, reward, next_state, action)
        if i % 1000 == 0:
            env2 = gym.make("FrozenLake-v0", is_slippery=False)
            env2.reset()
            current_state2 = 0
            pppp = SoftmaxPolicy(gq.theta, env, 1.0)
            N = 1000
            r = 0
            for iii in range(N):
                random_action = pppp.get_action(current_state2)
                new_state, reward, done, info = env2.step(random_action)
                if done:
                    env2.reset()
                    current_state2 = 0
                else:
                    current_state2 = new_state
                r += reward
            r /= 1000.0
            r_gq.append(np.max(r_gq + [np.copy(r)]))

        vrgq.update(current_state, reward, next_state, action)
        if i>batch_size:
            if i % 1000 == 0:
                env2 = gym.make("FrozenLake-v0", is_slippery=False)
                env2.reset()
                current_state2 = 0
                pppp = SoftmaxPolicy(vrgq.theta, env, 1.0)
                N = 1000
                r = 0
                for iii in range(N):
                    random_action = pppp.get_action(current_state2)
                    new_state, reward, done, info = env2.step(random_action)
                    if done:
                        env2.reset()
                        current_state2 = 0
                    else:
                        current_state2 = new_state
                    r += reward
                r /= 1000.0
                r_vrgq.append(np.max(r_vrgq + [np.copy(r)]))

        if (i + 1) % 10000 == 0:
            print("Current iteration:", i + 1, ". Time Spent:", time.time() - train_start)
            train_start = time.time()

        current_state = np.copy(next_state)
    print("Done")
    return r_pg, r_gq, r_vrgq

def easy_simulation(env, alpha, beta, batch_size, trajectory_length=50000, num_simulation=100, gamma=0.95, target=None):
    ini_start = time.time()

    print("Initialization...")
    ini_theta = np.random.normal(scale=12.0, size=env.num_features )
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
    pool = Pool(5)
    out = pool.starmap(_simulation, itertools.repeat(params, num_simulation))
    pool.close()
    pool.join()
    print("Training complete. Time Spent:", time.time() - train_start)
    return out


def main():
    np.random.seed(123)

    print("Set Up the Simulation Environment...")
    env = FrozenLake()
    print("Done.")

    gamma = 0.95
    # max_num_iteration = 50000
    # batch_size = 3000
    # alpha = 0.2
    # beta = 0.05


    max_num_iteration = 300000 # 500000
    batch_size = 3000
    alpha = 0.2
    beta = 0.1
    num_simulation = 5

    out = easy_simulation(env, alpha, beta, batch_size, trajectory_length=max_num_iteration,
                          num_simulation=num_simulation, gamma=gamma, target=None)

    return out


if __name__ == "__main__":
    out = main()
    r1 = []
    r2 = []
    r3 = []
    for i in out:
        r1.append(i[0])
        r2.append(i[1])
        r3.append(i[2])

    with open('reward-ac-raw-fl.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([r1, r2, r3], f)
    f.close()

    plt.figure()
    plt.plot(np.average(r1, axis=0),c="r")
    plt.plot(np.arange(0, 300001, 1000), np.average(r2, axis=0),c="b")
    plt.plot(np.arange(0, 297000, 1000), np.average(r3, axis=0),c="g")
    plt.show()


