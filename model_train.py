import numpy as np
import pandas as pd
import matplotlib as plt
import gym
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class TrainModel:
    
    def __init__(self,model):
        learning_parameter = 0.001
        Epochs = 500  # number of iterations 
        epsilon = 1.0 # How fast will learn 1 -> max but later modify according to learning with accracy
        epsilon_reduce = 0.995 # 99.5% reduced to previous epsilon
        gamma = 0.95
        target_model = clone_model(model)
        replay_buffer = deque(maxlen=20000)
        update_target_model = 10
        
    def greedy_epsilon_action(model,observation,epsilon):
    
    # Intialy not aware which action to perform so make a random action and then learn
        if np.random.random() > epsilon:
            predictions = model.predict(observation)
            action = np.argmax(predictions)  # picking max value to perform aaction

        else:
            #Explore
            action = np.random.randint(0, env.action_space.n) # choosing randomly an action b/w 0 to max actions can perform on env

        return action
    
    # Replay - Pick buffer size / sequence of frames/data to hold 
# use deque from collections 



    def replay(buffer,batch_size,model,target_model):

        if len(buffer) < batch_size:
            return # No action untill buffer filled
        samples = random.sample(buffer,batch_size)   # select ramdomly batch of values from buffer

        target_batch = []

        # samples -> collection of buffers ,1 buffer = 1 observations, observation -> state,reward,action, new_state,done_flag
        zipped_samples = list(zip(*samples))      #tuple of 
        states, actions ,rewards,new_states,dones = zipped_samples

        targets = target_model.predict(np.array(states))

        q_values = model.predict(np.array(new_states))

        for i in range(batch_size):

            q_value = q_values[i][0]
            target = targets[i].copy()

            if dones[i] :
                target[0][actions[i]] = rewards[i]

            else:
                target[0][actions[i]] = rewards[i] + q_value*gamma

            target_batch.append(target)

        model.fit(np.array(states),np.array(target_batch),epochs =1, verbose= False)
        

    def update_model_handler(ephocs, update_target_model, model,target_model):
        if ephocs >0 and ephocs % update_target_model == 0:
            target_model.set_weights(model.get_weights())

            
model.compile(loss='mse',optimizer=Adam(learning_rate=learning_parameter))            

score = 0

for ephocs in range(Epochs):
    observation = env.reset()[0]
    observation = observation.reshape([1,4])
    done = False
    
    point = 0 
    while not done:
        
        point += 1
        action = greedy_epsilon_action(model,observation,epsilon)
        next_observation,reward,done,truncate,info = env.step(action)
        next_observation = next_observation.reshape([1,4])
        
        replay_buffer.append((observation,action,reward,next_observation,done))
        observation = next_observation
        
        replay(replay_buffer,16,model,target_model)
    
    epsilon *= epsilon_reduce
    
    update_model_handler(ephocs, update_target_model, model,target_model)
    
    if point > score:
        score = point
        
    if ephocs % 10 == 0:
        print(f'score: {score} at ephoc:{ephocs} when epsilon: {epsilon}')
        
    
    
    

