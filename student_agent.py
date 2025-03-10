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
    global passenger_picked_up
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    state = ((obs[0], obs[1]) in stations, obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], passenger_picked_up)
    if state not in Q:
        action = random.randint(0, 5)
    else:
        action = np.argmax(Q[state])
    if action == 4 and ((obs[0], obs[1]) in stations) and obs[14]:
        passenger_picked_up = True
    if action == 5 and passenger_picked_up:
        passenger_picked_up = False
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.


with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)
passenger_picked_up = False