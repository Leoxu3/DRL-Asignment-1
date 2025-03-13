import numpy as np
import pickle
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv
from real_custom_taxi_env import RealTaxiEnv

def tabular_q_learning(episodes=25000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9999):
    def get_state(obs):
        obstacle_north = obs[10]
        obstacle_south = obs[11]
        obstacle_east = obs[12]
        obstacle_west = obs[13]

        return (obstacle_north, obstacle_south, obstacle_east, obstacle_west)
    
    env = RealTaxiEnv(fuel_limit=5000)
    q_table = {}
    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        state = get_state(obs)

        while not (done or truncated):
            if state not in q_table:
                q_table[state] = np.zeros(6)

            if np.random.rand() < epsilon:
                action = np.random.randint(6)
            else:
                action = np.argmax(q_table[state])

            obs, reward, done, truncated, _ = env.step(action)
            next_state = get_state(obs)

            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            if not done:
                best_next_q = np.max(q_table[next_state])
                q_table[state][action] += alpha * (reward + gamma * best_next_q - q_table[state][action])
            else:
                q_table[state][action] += alpha * (reward - q_table[state][action])

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
    
    return q_table, rewards_per_episode

print("Training agent...")
q_table, rewards_per_episode = tabular_q_learning()
print("Training completed.")
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)
plt.plot(rewards_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q Learning Training Progress")
plt.show()
print("Training completed and q-table saved.")
