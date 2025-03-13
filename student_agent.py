# Remember to adjust your student ID in meta.xml
import numpy as np
import random
import pickle

def get_action(obs):
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    if obstacle_north == 0:
        action = 1
    elif obstacle_south == 0:
        action = 0
    elif obstacle_east == 0:
        action = 2
    elif obstacle_west == 0:
        action = 3
    else:
        action = random.randint(4,5)
    return action