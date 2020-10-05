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

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
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

    dangerQ_critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    dangerQ_buff = ReplayBuffer(BUFFER_SIZE)

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    load_name = "sample_v0_40"
    print("Now we load the weight")
    try:
        actor.model.load_weights("saved/actormodel_{}.h5".format(load_name))
        critic.model.load_weights("saved/criticmodel_{}.h5".format(load_name))
        actor.target_model.load_weights("saved/actormodel_{}.h5".format(load_name))
        critic.target_model.load_weights("saved/criticmodel_{}.h5".format(load_name))

        dangerQ_critic.target_model.load_weights("saved/dangerQ_criticmodel_dangerQ_v3_99.h5".format(load_name))

        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    plt.figure()
    overall_scores = []
    model_name = "danger_Q_v1"

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 10) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ))
     
        total_reward = 0.
        cur_sample = []
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
            # if j == 91:
            #     print("adversarial attack!")
            #     if a_t[0][0] > 0:
            #         a_t[0][0] = -0.6
            #     else:
            #         a_t[0][0] = 0.6
            # # print("%.2f"%a_t[0][0])
            # a_t[0][2] += 0.7
            # if ob.speedX > 0.6:
                # a_t[0][1] = 0
            # if(step == 60):
                # a_t[0][0] = 1.0

            next_action_array = []
            for k in range(21):
                new_action = np.copy(a_t[0])
                new_action[0] = (k-10.0)/10.0
                next_action_array.append(new_action)
            next_action_array = np.array(next_action_array)
            cur_next_state_array = np.array([s_t for k in range(21)])
            
            dangerQ_value = dangerQ_critic.model.predict([cur_next_state_array, next_action_array])
            # maxQ = np.max(dangerQ_value, axis=0)[0]
            q_array = np.round(dangerQ_value[:,0], 3)
            # print(step, q_array[0], q_array[-1], q_array[int(a_t[0][0]*10+10)])
            cur_a_index = int(a_t[0][0]*10+10)
            # print(step, q_array[0]-q_array[cur_a_index], q_array[-1]-q_array[cur_a_index], q_array[int(a_t[0][0]*10+10)])
            print("{},{:.3f},{:.3f},{:.3f},{:.3f}".format(step, q_array[0]-q_array[cur_a_index], q_array[-1]-q_array[cur_a_index], q_array[int(a_t[0][0]*10+10)], a_t[0][0]))
            ob, r_t, done, info = env.step(a_t[0])
            # print "step: {} reward: {:.2f} action: {:.2f} {:.2f} {:.2f} ".format(j, r_t, a_t[0][0], a_t[0][1], a_t[0][2])
            # print "{:.5f} {:.5f} {:.5f} ".format(ob.angle, ob.trackPos, ob.speedX)

            # print "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(r_t, ob.speedX, ob.speedY, ob.speedZ, ob.rpm)
            # if(r_t < -50):
            #     r_t -= 10000
            #     done = True
            if j > 20 and ob.rpm <= 0.09426:
                r_t -= 1000
                done = True

            theta = 0.1
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ))
            # s_t1_new = np.array([val + np.abs(val)*random.uniform(-1,1)*theta for val in s_t1])
            # print(np.linalg.norm(s_t1_new - s_t1))
            # s_t1 = s_t1_new
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            cur_step_sample = [s_t.tolist(), a_t[0].tolist(), r_t, s_t1.tolist(), done]
            cur_sample.append(cur_step_sample)
            
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
            
            if j > 3000:
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
        s = "{},{},{:.3f}\n".format(i, j, total_reward)
        with open('logs/{}.csv'.format(model_name), 'a') as the_file:
            the_file.write(s)
        overall_scores.append(total_reward)
        plt.clf()
        plt.plot(overall_scores)
        plt.savefig("train_plots/{}_{}.jpg".format(model_name, int(step/10000)))
        with open('samples/{}_{:05d}.pk'.format(model_name, i), 'w') as outfile:
            pickle.dump(cur_sample, outfile)

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
    # playGame(1)
