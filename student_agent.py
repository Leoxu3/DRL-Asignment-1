# Remember to adjust your student ID in meta.xml
import numpy as np
import random
import pickle

def get_action(obs):
    global passenger_picked_up
    global q_table
    global pre_action

    def get_state(obs, passenger_picked_up, pre_action):
        taxi_loc = (obs[0], obs[1])
        stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        obstacle_north = obs[10]
        obstacle_south = obs[11]
        obstacle_east = obs[12]
        obstacle_west = obs[13]
        passenger_look = obs[14]
        destination_look = obs[15]

        station_north =False
        station_south = False
        station_east = False
        station_west = False
        station_middle = False
        for station in stations:
            if (taxi_loc[0] - 1, taxi_loc[1]) == station:
                station_north = True
            if (taxi_loc[0] + 1, taxi_loc[1]) == station:
                station_south = True
            if (taxi_loc[0], taxi_loc[1] + 1) == station:
                station_east = True
            if (taxi_loc[0], taxi_loc[1] - 1) == station:
                station_west = True
            if taxi_loc == station:
                station_middle = True

        return (obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, passenger_picked_up, station_north, station_south, station_east, station_west, station_middle, pre_action)

    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    state = get_state(obs, passenger_picked_up, pre_action)
    if state not in q_table:
        action = random.randint(0, 3)
    else:
        probs = softmax(q_table[state])
        action = np.random.choice(range(6), p=probs)

    if action == 4 and state[4] == 1 and state[11]:
        passenger_picked_up = True
    if action == 5 and state[5] == 1 and state[11] and passenger_picked_up:
        passenger_picked_up = False

    pre_action = action

    return action

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)
passenger_picked_up = False
pre_action = None