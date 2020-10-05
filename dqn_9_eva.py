from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.initializations import normal, identity
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, Flatten, Input, merge, Lambda
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
from collections import deque
import math

OU = OU()       #Ornstein-Uhlenbeck Process
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)


def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    BATCH = 32
    OBSERVATION = 50
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic
    REPLAY_MEMORY = 50000

    steering_num = 9
    acc_num = 1
    action_dim = steering_num * acc_num #Steering 5 * Acceleration/Brake 3
    state_dim = 24  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 500000.
    episode_count = 20000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    # epsilon = 1.0
    epsilon = 0.50
    # epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    print("Now we build the model")
    S = Input(shape=[state_dim])   
    h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
    h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
    # out = Dense(action_dim, activation='softmax',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
    out = Dense(action_dim, activation='linear',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
    model = Model(input=S,output=out)
    # print(model.summary())
    adam = Adam(lr=LRA)
    model.compile(loss='mse',optimizer=adam)
    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    D = deque()

    #Now load the weight
    # load_name = "sample_v0_40"
    # print("Now we load the weight")
    # try:
    #     print("Weight load successfully")
    # except:
    #     print("Cannot find the weight")
    model.load_weights("saved_dqn/dqn_9_v0_28.h5")
    # adam = Adam(lr=LRA)
    # model.compile(loss='mse',optimizer=adam)

    plt.figure()
    overall_scores = []
    model_name = "dqn_9_v0"

    def map_actions(a):
        a_index = np.argmax(a)
        steering_index = a_index % steering_num
        acc_index = a_index / steering_num
        steering = -1.0 + steering_index * 2.0 / (steering_num - 1)
        acc_brake = 0.0 + acc_index * 1.0 / (acc_num - 1)
        if a_index == 0:
            steering = -1.0
        if a_index == 1:
            steering = -0.50
        if a_index == 2:
            steering = -0.25
        if a_index == 3:
            steering = -0.10
        if a_index == 4:
            steering = 0.0
        if a_index == 5:
            steering = 0.10
        if a_index == 6:
            steering = 0.25
        if a_index == 7:
            steering = 0.5
        if a_index == 8:
            steering = 1.0
        
        # print(steering_index, acc_index)
        # if acc_brake >= 0:
        #     acc = acc_brake
        #     brake = 0.0
        # else:
        #     acc = 0
        #     brake = - acc_brake
        # brake /= 2.0
        # acc = acc_brake
        acc = 0
        brake = 0
        # print(a_index, acc_index, steering_index)
        a = [steering, acc, brake]
        return a
    
    # a_t = np.zeros([action_dim])
    # for i in range(action_dim):
    #     b_t = np.copy(a_t)
    #     b_t[i] = 1
    #     print(map_actions(b_t))
    # return

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        # print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 10) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ))
     
        total_reward = 0.
        cur_sample = []
        cur_diffs = []
        attacked_steps = 0
        beta = 0.910 - i/1000.0
        for j in range(max_steps):
            # if j == 50:
                # time.sleep(0.099)
                # continue
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([action_dim])
            if train_indicator and random.random() <= epsilon:
                # print("----------Random Action----------")
                action_index = random.randrange(action_dim)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t.reshape(1, s_t.shape[0]))       #input a stack of 4 images, get the prediction
                # print(q)
                max_Q = np.argmax(q)
                action_index = max_Q
                # print(q, action_index)
                if(math.isnan(q[0][0])):
                    return
                a_t[max_Q] = 1
            
            
            new_at = map_actions(a_t)
            target_speed = 0.50
            speed_p = 3.0
            if ob.speedX < target_speed:
                new_at[1] = (target_speed - ob.speedX) * speed_p
            else:
                new_at[1] = 0.1
            # if j < 20 and train_indicator:
            #     new_at[1] += 0.5
            #     new_at[2] = 0
            # if(step == 90):
                # new_at[0] = -1.0

            a_sm = softmax(q[0])
            diff = np.max(a_sm) - np.min(a_sm)
            cur_diffs.append(diff)
            if diff > beta:
                adv_q = np.argmin(q)
                a_t = np.zeros([action_dim])
                a_t[adv_q] = 1
                new_at = map_actions(a_t)
                attacked_steps += 1

            a_t = new_at
            ob, r_t, done, info = env.step(new_at)
            print "step: {} reward: {:.5f} action: {:.5f} {:.5f} {:.5f} ".format(j, r_t, a_t[0], a_t[1], a_t[2])

            if j > 10 and ob.rpm <= 0.09426:
                r_t -= 1000
                done = True

            theta = 0.1
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ))
            
            # buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            # cur_step_sample = [s_t.tolist(), a_t[0].tolist(), r_t, s_t1.tolist(), done]
            # cur_sample.append(cur_step_sample)
            terminal = done
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
       
            if (train_indicator and step > OBSERVATION):
                minibatch = random.sample(D, BATCH)

                #Now we do the experience replay
                state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
                # print(len(state_t))
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                # print(state_t1.shape)
                # print(state_t1)
                # print(state_t1.reshape(32, 24).shape)
                state_t = state_t.reshape(32, 24)
                targets = model.predict(state_t)
                state_t1 = state_t1.reshape(32, 24)
                Q_sa = model.predict(state_t1)
                targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

                loss += model.train_on_batch(state_t, targets)

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
                model.save_weights("saved_dqn/{}_{}.h5".format(model_name, int(step/10000)), overwrite=True)
                # actor.model.save_weights("saved/actormodel_{}_{}.h5".format(model_name, int(step/10000)), overwrite=True)
                # with open("actormodel.json", "w") as outfile:
                #     json.dump(actor.model.to_json(), outfile)

                # critic.model.save_weights("saved/criticmodel_{}_{}.h5".format(model_name, int(step/10000)), overwrite=True)
                # with open("criticmodel.json", "w") as outfile:
                #     json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        s = "{},{},{},{},{:.3f}\n".format(i, beta, attacked_steps, j, total_reward)
        # s = "{},{},{:.3f}\n".format(i, j, total_reward)
        with open('logs/{}_tactics.csv'.format(model_name), 'a') as the_file:
            the_file.write(s)
        overall_scores.append(total_reward)
        # plt.clf()
        # plt.plot(overall_scores)
        # plt.savefig("train_plots/{}_{}.jpg".format(model_name, int(step/10000)))
        # with open('samples/{}_{:05d}.pk'.format(model_name, i), 'w') as outfile:
            # pickle.dump(cur_sample, outfile)

        # plt.clf()
        # plt.plot(cur_diffs)
        # plt.savefig("cur_diff_new.jpg")
        

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
    # playGame(1)
