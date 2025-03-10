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
    if obs[10] == 0:
        return 1
    elif obs[11] == 0:
        return 0
    elif obs[12] == 0:
        return 2
    elif obs[13] == 0:
        return 3
    else:
        return random.randint(0, 3)
    # You can submit this random agent to evaluate the performance of a purely random strategy.