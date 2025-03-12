import numpy as np
import pickle
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv
from real_custom_taxi_env import RealTaxiEnv

def policy_learning(episodes=6000, alpha=0.005, gamma=0.99):
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
    
    env = RealTaxiEnv(fuel_limit=1000)
    policy_table = {}
    rewards_per_episode = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        passenger_picked_up = False
        pre_action = None
        state = get_state(obs, passenger_picked_up, pre_action)
        trajectory = []

        while not (done or truncated):
            if state not in policy_table:
                policy_table[state] = np.zeros(6)

            logits = policy_table[state]
            action_probs = softmax(logits)
            action = np.random.choice(range(6), p=action_probs)

            if action == 4 and state[4] == 1 and state[5]:
                passenger_picked_up = True
            if action == 5 and state[6] == 1 and state[5]:
                passenger_picked_up = False

            pre_action = action

            obs, reward, done, truncated, _ = env.step(action)
            next_state = get_state(obs, passenger_picked_up, pre_action)

            trajectory.append((state, action, reward))

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        G = 0
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = reward + gamma * G

            logits = policy_table[state]
            probs = softmax(logits)
            grad = -probs
            grad[action] += 1
            policy_table[state] += alpha * G * grad

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")
    
    return policy_table, rewards_per_episode

print("Training agent...")
policy_table, rewards_per_episode = policy_learning()
print("Training completed.")
with open("policy.pkl", "wb") as f:
    pickle.dump(policy_table, f)
plt.plot(rewards_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Policy Learning Training Progress")
plt.show()
print("Training completed and policy saved.")
