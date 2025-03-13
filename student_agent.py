# Remember to adjust your student ID in meta.xml
import numpy as np
import random
import pickle

def get_action(obs):
    def get_state(obs):
        obstacle_north = obs[10]
        obstacle_south = obs[11]
        obstacle_east = obs[12]
        obstacle_west = obs[13]

        return (obstacle_north, obstacle_south, obstacle_east, obstacle_west)
    
    state = get_state(obs)
    if state not in q_table:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(q_table[state])
    return action

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)