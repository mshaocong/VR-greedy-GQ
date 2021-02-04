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
    train_start = time.time()
    env.reset()
    current_state = env.current_state

    gq = GreedyGQ_Base(env, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    gq.set_theta(ini_theta)

    vrgq = VRGreedyGQ(env, batch_size=batch_size, target_policy=target, eta_theta=alpha, eta_omega=beta, gamma=gamma)
    vrgq.set_theta(ini_theta)

    gq_hist_avg = [evaluate_J(env, gq.theta)]
    vrgq_hist_avg = [evaluate_J(env, gq.theta)]

    tmpp_env = env.get_copy()
    pppp = SoftmaxPolicy(gq.theta, tmpp_env, 1.0)
    ppp = np.zeros((tmpp_env.num_state, tmpp_env.num_action))
    for ssss in tmpp_env.state_space:
        for aaaa in tmpp_env.action_space:
            ppp[ssss,aaaa] = pppp.policy(aaaa, ssss)
    tmpp_env.set_behavior_policy(ppp)
    stat = compute_stationary_dist(tmpp_env.trans_kernel)
    r = get_total_reward(reward=tmpp_env.reward, stationary_dist=stat)
    r_gq = [np.copy(r)]
    r_vrgq = [np.copy(r)]

    aaaaaaa = evaluate_J_obj(env, gq.theta)
    obj_gq = [np.copy(aaaaaaa)]
    obj_vrgq = [np.copy(aaaaaaa)]
    count = 1

    best_gq_obj = np.copy(aaaaaaa)
    gq_best_theta = np.copy(gq.theta)
    best_vrgq_obj = np.copy(aaaaaaa)
    vrgq_best_theta = np.copy(gq.theta)
    for i in range(trajectory_length):
        next_state, reward, action = env.step()

        gq.update(current_state, reward, next_state, action)
        if i % 10 == 0:
            tmpp_env = env.get_copy()
            pppp = SoftmaxPolicy(gq.theta, tmpp_env, 1.0)
            ppp = np.zeros((tmpp_env.num_state, tmpp_env.num_action))
            for ssss in tmpp_env.state_space:
                for aaaa in tmpp_env.action_space:
                    ppp[ssss,aaaa] = pppp.policy(aaaa, ssss)
            tmpp_env.set_behavior_policy(ppp)
            stat = compute_stationary_dist(tmpp_env.trans_kernel)
            r = get_total_reward(reward=tmpp_env.reward, stationary_dist=stat)
            r_gq.append(np.max(r_gq + [np.copy(r)]) )

            gq_hist_avg.append( np.min(gq_hist_avg + [evaluate_J(env, gq.theta[0])]) )

            obj_tmp = evaluate_J_obj(env, gq.theta.transpose() )[0]
            if obj_tmp < best_gq_obj:
                best_gq_obj = obj_tmp
                vrgq_best_theta = gq.theta[0]
            if i > 100:
                obj_gq.append(np.min( obj_gq+[np.copy(obj_tmp)] ) )

            #gq_hist_avg.append( evaluate_J(env, gq.theta[0]) )
            #gq_hist_avg.append(
            #    (len(gq_hist_avg) * gq_hist_avg[-1] + evaluate_J(env, gq.theta[0])) / (len(gq_hist_avg) + 1))

        vrgq.update(current_state, reward, next_state, action)
        if i>batch_size:
            if i % 10 == 0:
                tmpp_env = env.get_copy()
                pppp = SoftmaxPolicy(vrgq.theta, tmpp_env, 1.0)
                ppp = np.zeros((tmpp_env.num_state, tmpp_env.num_action))
                for ssss in tmpp_env.state_space:
                    for aaaa in tmpp_env.action_space:
                        ppp[ssss,aaaa] = pppp.policy(aaaa, ssss)
                tmpp_env.set_behavior_policy(ppp)
                stat = compute_stationary_dist(tmpp_env.trans_kernel)
                r = get_total_reward(reward=tmpp_env.reward, stationary_dist=stat)
                r_vrgq.append(np.max(r_vrgq + [np.copy(r)]) )
                #vrgq_hist_avg.append((len(vrgq_hist_avg)  * vrgq_hist_avg[-1] + evaluate_J(env, vrgq.theta)) /(len(vrgq_hist_avg) + 1))
                vrgq_hist_avg.append(np.min(vrgq_hist_avg + [evaluate_J(env, vrgq.theta)]))
                #vrgq_hist_avg.append( evaluate_J(env, vrgq.theta))

                obj_tmp = evaluate_J_obj(env, vrgq.theta)
                if obj_tmp < best_gq_obj:
                    best_vrgq_obj = obj_tmp
                    vrgq_best_theta = vrgq.theta

                if i  > 100:
                    obj_vrgq.append(np.min( obj_vrgq+[np.copy(obj_tmp)] ) )

        if (i + 1) % 10000 == 0:
            print("Current iteration:", i + 1, ". Time Spent:", time.time() - train_start)
            train_start = time.time()
        count += 1

        current_state = np.copy(next_state)
    return gq_hist_avg, vrgq_hist_avg, r_gq, r_vrgq, obj_gq, obj_vrgq
    # return gq_hist_last, vrgq_hist_last

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
    pool = Pool(4)
    out = pool.starmap(_simulation, itertools.repeat(params, num_simulation))
    pool.close()
    pool.join()
    print("Training complete. Time Spent:", time.time() - train_start)
    return out


def main():
    np.random.seed(123)

    num_states = 5
    num_actions = 3
    branching_factor = 2
    num_features = 4

    print("Set Up the Simulation Environment...")
    env = Garnet(num_states, num_actions, branching_factor, num_features)
    print("Done.")

    gamma = 0.95
    # max_num_iteration = 50000
    # batch_size = 3000
    # alpha = 0.2
    # beta = 0.05


    max_num_iteration = 50000
    batch_size = 3000
    alpha = 0.02
    beta = 0.01
    num_simulation = 32 #16

    out = easy_simulation(env, alpha, beta, batch_size, trajectory_length=max_num_iteration,
                          num_simulation=num_simulation, gamma=gamma, target=None)

    return out


if __name__ == "__main__":
    out = main()

    hist_gq = []
    hist_vrgq = []
    r1 = []
    r2 = []
    gq_obj = []
    vrgq_obj = []
    for output in out:
        hist_gq.append(output[0])
        hist_vrgq.append(output[1])
        r1.append(output[2])
        r2.append(output[3])
        gq_obj.append(output[4])
        vrgq_obj.append(output[5])

    with open('reward-gq-alpha002-beta001-bs3000-seed123-max.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(r1, f)
    f.close()
    with open('reward-vrgq-alpha002-beta001-bs3000-seed123-max.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(r2, f)
    f.close()
    with open('obj-gq-alpha002-beta001-bs3000-seed123-min.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(gq_obj, f)
    f.close()
    with open('obj-vrgq-alpha002-beta001-bs3000-seed123-min.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(vrgq_obj, f)
    f.close()

    plt.figure()
    plt.plot(np.average(r1, axis=0),c="r")
    plt.plot(np.average(r2, axis=0),c="b")
    plt.show()

    plt.figure()
    plt.plot(np.average(gq_obj, axis=0),c="r")
    plt.plot(np.average(vrgq_obj, axis=0),c="b")
    #plt.ylim(0,10)
    plt.show()

