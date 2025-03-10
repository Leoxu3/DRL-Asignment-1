# train_agent.py

import numpy as np
import pickle
import random
import gym
import os
from simple_custom_taxi_env import SimpleTaxiEnv
from real_custom_taxi_env import RealTaxiEnv

q_table = {}

def tabular_q_learning(episodes=100000, alpha=0.75, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99999):
    env = RealTaxiEnv(fuel_limit=5000)
    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        taxi_loc = (obs[0], obs[1])
        state = ((taxi_loc in stations), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])

        while not done and not truncated:
            if state not in q_table:
                q_table[state] = np.zeros(6)

            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                action = np.argmax(q_table[state])

            next_obs, reward, done, truncated, info = env.step(action)
            next_taxi_loc = (next_obs[0], next_obs[1])
            next_state = ((next_taxi_loc in stations), next_obs[10], next_obs[11], next_obs[12], next_obs[13], next_obs[14], next_obs[15])
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon * decay_rate, epsilon_end)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")

print("Training agent...")
tabular_q_learning()
print("Training completed.")
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)
print("Training completed and Q-table saved.")
