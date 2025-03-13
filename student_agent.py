# Remember to adjust your student ID in meta.xml
import numpy as np
import random
import pickle

def get_action(obs):
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    if obstacle_north != 1:
        return 1
    if obstacle_east != 1:
        return 2
    if obstacle_south != 1:
        return 0
    if obstacle_west != 1:
        return 3
    else:
        return random.randint(4,5)