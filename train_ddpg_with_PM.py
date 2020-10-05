from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
from keras.models import load_model


# x = np.array([ 4.82767379e-01,  5.92105016e-02,  3.61700505e-01,  2.74807483e-01,
#   2.31401995e-01,  2.07236990e-01,  1.95800006e-01,  1.89892501e-01,
#   1.84837490e-01,  1.81293502e-01,  1.77807003e-01,  1.74377009e-01,
#   1.71005994e-01,  1.66384503e-01,  1.61247000e-01,  1.52030498e-01,
#   1.35238498e-01,  1.11962005e-01,  8.79574940e-02,  4.76383008e-02,
#   4.78339800e-01,  6.97819047e-01,  4.60800716e-01,  5.00754069e-01,
#  -1.00000000e+00,  9.99979496e-01,  8.71338917e-13])
# x_s = np.array([x, x])
# pre_y = pre_model.predict(x_s)
# print(pre_y[0])

OU = OU()       #Ornstein-Uhlenbeck Process

def restore_states(s_t_scaled):
    s_t_s = np.copy(s_t_scaled)
    for s_t in s_t_s:
        s_t[0] = restore_data(s_t[0], 0.5)
        s_t[20] = restore_data(s_t[20], 2.5)
        s_t[21] = restore_data(s_t[21], 0.7)
        s_t[22] = restore_data(s_t[22], 0.7)
        s_t[23] = restore_data(s_t[23], 0.7)
    return s_t_s

def restore_state(s_t_scaled):
    s_t = np.copy(s_t_scaled)
    s_t[0] = restore_data(s_t[0], 0.5)
    s_t[20] = restore_data(s_t[20], 2.5)
    s_t[21] = restore_data(s_t[21], 0.7)
    s_t[22] = restore_data(s_t[22], 0.7)
    s_t[23] = restore_data(s_t[23], 0.7)
    return s_t

def rescale_states(s_t):
    s_t_scaled_s = np.copy(s_t)
    for s_t_scaled in s_t_scaled_s:
        s_t_scaled[0] = rescale_data(s_t_scaled[0], 0.5)
        s_t_scaled[20] = rescale_data(s_t_scaled[20], 2.5)
        s_t_scaled[21] = rescale_data(s_t_scaled[21], 0.7)
        s_t_scaled[22] = rescale_data(s_t_scaled[22], 0.7)
        s_t_scaled[23] = rescale_data(s_t_scaled[23], 0.7)
    return s_t_scaled_s

def rescale_state(s_t):
    s_t_scaled = np.copy(s_t)
    s_t_scaled[0] = rescale_data(s_t_scaled[0], 0.5)
    s_t_scaled[20] = rescale_data(s_t_scaled[20], 2.5)
    s_t_scaled[21] = rescale_data(s_t_scaled[21], 0.7)
    s_t_scaled[22] = rescale_data(s_t_scaled[22], 0.7)
    s_t_scaled[23] = rescale_data(s_t_scaled[23], 0.7)
    return s_t_scaled

def restore_data(val, scale):
    return 2*scale*val - scale

def rescale_data(val, scale):
    return (val + scale) / (scale * 2)

def calsulate_d(st):
    return (st[20]*2.5*2) - 2.5

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    time.sleep(1)
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 24  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 300000.
    episode_count = 20000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1.0
    # epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
    pre_model = load_model("weights_rescale_all-0000.hdf5")
    # x = np.array([ 4.82767379e-01,  5.92105016e-02,  3.61700505e-01,  2.74807483e-01,
    #     2.31401995e-01,  2.07236990e-01,  1.95800006e-01,  1.89892501e-01,
    #     1.84837490e-01,  1.81293502e-01,  1.77807003e-01,  1.74377009e-01,
    #     1.71005994e-01,  1.66384503e-01,  1.61247000e-01,  1.52030498e-01,
    #     1.35238498e-01,  1.11962005e-01,  8.79574940e-02,  4.76383008e-02,
    #     4.78339800e-01,  6.97819047e-01,  4.60800716e-01,  5.00754069e-01,
    #     -1.00000000e+00,  9.99979496e-01,  8.71338917e-13])
    # x_s = np.array([x, x])
    # pre_y = pre_model.predict(x_s)
    # print(x_s[0])
    # print(pre_y[0])

    #Now load the weight
    load_name = "sample_v0_40"
    print("Now we load the weight")
    try:
        actor.model.load_weights("saved/actormodel_{}.h5".format(load_name))
        critic.model.load_weights("saved/criticmodel_{}.h5".format(load_name))
        actor.target_model.load_weights("saved/actormodel_{}.h5".format(load_name))
        critic.target_model.load_weights("saved/criticmodel_{}.h5".format(load_name))
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    plt.figure()
    overall_scores = []
    model_name = "sample_v0"

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ))
     
        total_reward = 0.
        cur_sample = []
        attack_valid = 1
        gap = (i/10)/100.0
        attack_step = -1
        attack_target = 0
        for j in range(max_steps):
            # if j == 50:
                # time.sleep(0.099)
                # continue
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            # if j > 120:
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            if j < 20 and train_indicator:
                a_t[0][1] += 0.5
            # if j == 71:
            #     print("cp attack!")
            #     if a_t[0][0] > 0:
            #         a_t[0][0] = -0.3
            #     else:
            #         a_t[0][0] = 0.3
            # print("%.2f"%a_t[0][0])
            # a_t[0][2] += 0.7
            # if ob.speedX > 0.6:
                # a_t[0][1] = 0
            # if(step == 60):
                # a_t[0][0] = 1.0
            s_t_scaled = rescale_state(s_t)
            # print(s_t[0])
            s_t_0 = restore_state(s_t_scaled)
            # print(s_t_0[0])
            new_a_t = actor.model.predict(s_t_0.reshape(1, s_t_0.shape[0]))
            s_t_scaled_list = np.array([np.copy(s_t_scaled) for val in range(21)])
            actions = np.array([np.copy(a_t[0]) for val in range(21)])
            for val in range(21):
                actions[val][0] = -1.0 + val/10.0
            # print(actions)
            x_0 = np.hstack((s_t_scaled_list, actions))
            # print(x_0.shape, s_t_scaled_list.shape, actions.shape)
            pre_y = pre_model.predict(x_0)
            # print(x_0[0])
            # print(pre_y[0])
            

            steer_index = int(a_t[0][0]*10.0 + 10.0)
            for pre_step in range(2):
                restore_new_Y = restore_states(pre_y)
                actions = actor.model.predict(restore_new_Y)
                x_step1 = np.hstack((pre_y, actions))
                pre_y = pre_model.predict(x_step1)

            for index in range(21):
                diff = calsulate_d(pre_y[index]) - calsulate_d(pre_y[steer_index])
                pro = np.random.random()
                if diff > gap and attack_valid == 1 and pro > 0.8 and j > 50:
                    a_t[0][0] = -1.0 + index/10.0
                    print("adv!", diff, "pro:", pro)
                    attack_step = j
                    attack_target = a_t[0][0]
                    attack_valid -= 1


            # dis_list = np.array([(calsulate_d(st) - calsulate_d(pre_y[steer_index])) for st in pre_y])
            # print("{:.2f}".format(max(dis_list)*100000))
            # print("{}".format(max(dis_list)*100000))

            # s_t_scaled = np.copy(s_t1)
            # s_t_scaled[0] = rescale_data(s_t_scaled[0], 0.5)
            # s_t_scaled[20] = rescale_data(s_t_scaled[20], 2.5)
            # s_t_scaled[21] = rescale_data(s_t_scaled[21], 0.7)
            # s_t_scaled[22] = rescale_data(s_t_scaled[22], 0.7)
            # s_t_scaled[23] = rescale_data(s_t_scaled[23], 0.7)
            # actions = actor.model.predict(s_t_scaled.reshape(1, s_t_scaled.shape[0]))
            # print(actions[0][0])

            # ob, r_t, done, info = env.step(new_a_t[0])
            ob, r_t, done, info = env.step(a_t[0])
            # print "step: {} reward: {:.5f} action: {:.5f} {:.5f} {:.5f} ".format(j, r_t, a_t[0][0], a_t[0][1], a_t[0][2])
            # print(a_t[0][0])

            # print "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(r_t, ob.speedX, ob.speedY, ob.speedZ, ob.rpm)
            # if(r_t < -50):
            #     r_t -= 10000
            #     done = True
            if j > 20 and ob.rpm <= 0.09426:
                r_t -= 1000
                done = True

            theta = 0.1
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ))

            # action_states = []
            # for i in range(-5, 6):

            # s_t1_new = np.array([val + np.abs(val)*random.uniform(-1,1)*theta for val in s_t1])
            # print(np.linalg.norm(s_t1_new - s_t1))
            # s_t1 = s_t1_new
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            # cur_step_sample = [s_t.tolist(), a_t[0].tolist(), r_t, s_t1.tolist(), done]
            # cur_sample.append(cur_step_sample)
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break
            
            if j > 500:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("saved/actormodel_{}_{}.h5".format(model_name, int(step/10000)), overwrite=True)
                # with open("actormodel.json", "w") as outfile:
                #     json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("saved/criticmodel_{}_{}.h5".format(model_name, int(step/10000)), overwrite=True)
                # with open("criticmodel.json", "w") as outfile:
                #     json.dump(critic.model.to_json(), outfile)
        
        
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        s = "{},{},{},{},{},{:.3f}\n".format(gap, attack_step, attack_target, i, j, total_reward)
        attack_valid = 1
        attack_step = -1
        attack_target = 0
        with open('logs/pm_adv_test.csv'.format(model_name), 'a') as the_file:
            the_file.write(s)
        overall_scores.append(total_reward)
        plt.clf()
        plt.plot(overall_scores)
        plt.savefig("train_plots/{}_{}.jpg".format(model_name, int(step/10000)))
        # with open('samples/{}_{:05d}.pk'.format(model_name, i), 'w') as outfile:
            # pickle.dump(cur_sample, outfile)

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
    # playGame(1)
    pass

