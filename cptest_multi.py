from gym_torcs_multi import TorcsEnv
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
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool, Process, Semaphore, Lock

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(env_num, lock, attack_step):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 24  #of sensors input

    np.random.seed(1337 + env_num)

    vision = False

    EXPLORE = 100000.
    episode_count = 1000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1.0
    # epsilon = 1
    indicator = 0
    collect_indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    # buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # print(actor.model.summary())
    # return

    # Generate a Torcs environment
    time.sleep(4 * env_num)
    env = TorcsEnv(env_num, lock,vision=vision, throttle=True,gear_change=False)
    

    load_name = "sample_v0_40"
    # print("Now we load the weight")
    try:
        actor.model.load_weights("saved/actormodel_{}.h5".format(load_name))
        critic.model.load_weights("saved/criticmodel_{}.h5".format(load_name))
        actor.target_model.load_weights("saved/actormodel_{}.h5".format(load_name))
        critic.target_model.load_weights("saved/criticmodel_{}.h5".format(load_name))
        # print("Weight load successfully")
    except:
        print("Cannot find the weight")

    plt.figure()
    overall_scores = []
    model_name = "multi_cp_300_v2"
    
    attacks = []
    for i in range(-100, 101):
        val = i/100.0
        attacks.append([attack_step, val])
    time.sleep(4 * (16 - env_num))
    print("env_num:{} TORCS Data Collection Start.".format(env_num))
    for i in range(len(attacks)):

        # print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        # if np.mod(i, 3) == 0:
        #     ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        # else:
            # ob = env.reset()
        # if step > 2000:
        #     break
        ob = env.reset()
        
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ))
     
        total_reward = 0.
        cur_sample = []
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            # if j > 120:
            noise_t[0][0] = collect_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = collect_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = collect_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            if j < 10 :
                a_t[0][1] += 0.5
            # print("%.2f"%a_t[0][0])
            # a_t[0][2] += 0.7
            if(j == attacks[i][0]):
                print('cp attack on {} with {}'.format(attacks[i][0], attacks[i][1]))
                a_t[0][0] = attacks[i][1]
            
            ob, r_t, done, info = env.step(a_t[0])
            # print "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(j, r_t, a_t[0][0], a_t[0][1], a_t[0][2])
            # print(ob.track)
            # print(ob.trackPos)
            # print "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(r_t, ob.speedX, ob.speedY, ob.speedZ, ob.rpm)
            # if(r_t < -50):
            #     r_t -= 10000
            #     done = True
            if j > 20 and ob.rpm <= 0.09426:
                r_t -= 1000
                done = True

            # print(ob.angle.shape, ob.track.shape, ob.trackPos.shape, ob.speedX.shape, ob.speedY.shape, ob.speedZ.shape, ob.wheelSpinVel.shape, ob.rpm.shape)
            # print(ob.speedX*300, ob.speedY*300, ob.speedZ*300, ob.angle*3.1416)
            
            theta = 0.0001
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ))
            # s_t1 = np.array([val+np.abs(val)*theta*random.uniform(-1, 1) for val in s_t1])
            # s_t1 = np.hstack((new_angle, ob.track, new_trackPos, new_speedX, new_speedY, new_speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            # buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            cur_step_sample = [s_t.tolist(), a_t[0].tolist(), r_t, s_t1.tolist(), done]
            cur_sample.append(cur_step_sample)
            
            #Do the batch update
            # batch = buff.getBatch(BATCH_SIZE)
            # states = np.asarray([e[0] for e in batch])
            # actions = np.asarray([e[1] for e in batch])
            # rewards = np.asarray([e[2] for e in batch])
            # new_states = np.asarray([e[3] for e in batch])
            # dones = np.asarray([e[4] for e in batch])
            # y_t = np.asarray([e[1] for e in batch])

            # target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            # for k in range(len(batch)):
            #     if dones[k]:
            #         y_t[k] = rewards[k]
            #     else:
            #         y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            # if (train_indicator):
            #     loss += critic.model.train_on_batch([states,actions], y_t) 
            #     a_for_grad = actor.model.predict(states)
            #     grads = critic.gradients(states, a_for_grad)
            #     actor.train(states, grads)
            #     actor.target_train()
            #     critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break
            
            if j > 300:
                break

        # if np.mod(i, 3) == 0:
        #     if (train_indicator):
        #         print("Now we save model")
        #         actor.model.save_weights("saved/actormodel_{}_{}.h5".format(model_name, int(step/10000)), overwrite=True)
        #         with open("actormodel.json", "w") as outfile:
        #             json.dump(actor.model.to_json(), outfile)

        #         critic.model.save_weights("saved/criticmodel_{}_{}.h5".format(model_name, int(step/10000)), overwrite=True)
        #         with open("criticmodel.json", "w") as outfile:
        #             json.dump(critic.model.to_json(), outfile)
        print("env_num: {} episode: {} total_step: {} total_reward: {}".format(env_num, i, step, total_reward))
        # print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward) + "Total Step: " + str(step))
        s = "{},{},{},{:.3f}\n".format(attacks[i][0], attacks[i][1], j, total_reward)
        with open('logs/{}.csv'.format(model_name), 'a') as the_file:
            the_file.write(s)
        # overall_scores.append(total_reward)
        # plt.clf()
        # plt.plot(overall_scores)
        # plt.savefig("train_plots/{}_{}_{}.jpg".format(model_name, env_num, int(step/10000)))
        # with open("samples_multi/{}_{}_{:04d}_{:05d}.pk".format(model_name, env_num, epoch, i), 'w') as outfile:
        #     pickle.dump(cur_sample, outfile)

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    # playGame()
    os.system('pkill torcs')
    os.environ["TORCS_RESTART"] = "0"
    # playGame(0)
    thread_num = 16
    # envs = range(thread_num)
    # thread_pool = Pool(thread_num)
    # thread_pool.map(playGame, envs)
    lock = Lock()
    process_pool = []
    START = 30

    for epoch in range(20):
        for i in range(thread_num):
            attack_step = START + epoch * thread_num + i
            # print(attack_step)
            pro = Process(target = playGame, args=(i, lock, attack_step, ))
            pro.start()
            process_pool.append(pro)

        for pro in process_pool:
            pro.join()
