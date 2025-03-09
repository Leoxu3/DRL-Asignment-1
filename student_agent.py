# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    with open("q_table.pkl", "rb") as f:
        Q = pickle.load(f)
    taxi_loc = (obs[0], obs[1])
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])] 
    if taxi_loc in stations:
        state = (True, obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])
    else:
        state = (False, obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])
    if state not in Q:
        return random.randint(0, 5)
    return int(np.argmax(Q[state]))
    # You can submit this random agent to evaluate the performance of a purely random strategy.

