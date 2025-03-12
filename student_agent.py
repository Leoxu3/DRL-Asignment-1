# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random

def get_action(obs):
    global passenger_picked_up
    global q_table
    global pre_action
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    def get_state(obs, passenger_picked_up, pre_action):
        taxi_loc = (obs[0], obs[1])
        stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        obstacle_north = obs[10]
        obstacle_south = obs[11]
        obstacle_east = obs[12]
        obstacle_west = obs[13]
        passenger_look = obs[14]
        destination_look = obs[15]

        return (obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, (taxi_loc in stations), destination_look, passenger_picked_up, pre_action)
    
    state = get_state(obs, passenger_picked_up, pre_action)
    if state not in q_table:
        action = random.randint(0, 5)
    else:
        if np.random.rand() < 0.1:
            action = np.random.randint(6)
        else:
            action = np.argmax(q_table[state])

    if action == 4 and state[4] == 1 and state[5]:
        passenger_picked_up = True
    if action == 5 and state[6] == 1 and state[5]:
        passenger_picked_up = False

    pre_action = action

    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.


with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)
passenger_picked_up = False
pre_action = None