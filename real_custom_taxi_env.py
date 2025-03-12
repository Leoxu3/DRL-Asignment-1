import random
import importlib.util
import time
from IPython.display import clear_output

class RealTaxiEnv():
    def __init__(self, fuel_limit=5000):
        self.grid_size = 0
        self.fuel_limit = fuel_limit
        self.current_fuel = 0
        self.obstacles = set()
        self.taxi_loc = None
        self.stations = None
        self.passenger_loc = None
        self.destination = None
        self.passenger_picked_up = False
    def reset(self):
        self.grid_size = random.randint(5, 10)
        self.current_fuel = self.fuel_limit
        all_locations = set((i, j) for i in range(self.grid_size) for j in range(self.grid_size))
        self.obstacles = set(random.sample(list(all_locations), random.randint(0, 0)))
        all_locations -= self.obstacles
        self.taxi_loc = random.choice(list(all_locations))
        self.stations = random.sample(list(all_locations), 4)
        #self.stations = [(0,0), (0, self.grid_size-1), (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]
        self.stations = list(self.stations)
        self.passenger_loc, self.destination = random.sample(self.stations, 2)
        self.passenger_picked_up = False  
        return self.get_state(), {}
    def get_state(self):
        taxi_row, taxi_col = self.taxi_loc
        obstacle_north = int(taxi_row == 0 or ((taxi_row - 1, taxi_col) in self.obstacles))
        obstacle_south = int(taxi_row == self.grid_size - 1 or ((taxi_row + 1, taxi_col) in self.obstacles))
        obstacle_east  = int(taxi_col == self.grid_size - 1 or ((taxi_row, taxi_col + 1) in self.obstacles))
        obstacle_west  = int(taxi_col == 0 or ((taxi_row, taxi_col - 1) in self.obstacles))
        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int((taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle
        state = (taxi_row, taxi_col, self.stations[0][0], self.stations[0][1],self.stations[1][0], self.stations[1][1], self.stations[2][0], self.stations[2][1], self.stations[3][0], self.stations[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def step(self, action):
        """Perform an action and update the environment state."""
        self.current_fuel -= 1
        reward = 0
        done = False
        truncated = False

        if action in [0, 1, 2, 3]:  
            reward -= 0.1

            next_row, next_col = self.taxi_loc
            if action == 0 :  # Move South
                next_row += 1
            elif action == 1:  # Move North
                next_row -= 1
            elif action == 2 :  # Move East
                next_col += 1
            elif action == 3 :  # Move West
                next_col -= 1

            if (not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size)) or ((next_row, next_col) in self.obstacles):
                reward -= 5
            else:
                self.taxi_loc = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = (next_row, next_col)
        elif action == 4:  
            if (self.taxi_loc == self.passenger_loc) and (not self.passenger_picked_up):
                self.passenger_picked_up = True
                reward += 100
            else:
                reward -= 10
        elif action == 5:
            if (self.taxi_loc == self.destination) and self.passenger_picked_up:
                reward += 500
                done = True
            else:
                reward -= 10
        
        if self.current_fuel <= 0:
            truncated = True
            reward -= 10

        return self.get_state(), reward, done, truncated, {}
    
    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        
        grid[self.stations[0][0]][self.stations[0][1]] = 'R'
        grid[self.stations[1][0]][self.stations[1][1]] = 'G'
        grid[self.stations[2][0]][self.stations[2][1]] = 'Y'
        grid[self.stations[3][0]][self.stations[3][1]] = 'B'
        
        py, px = self.passenger_loc
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'

        dy, dx = self.destination
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'

        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        print(f"\nStep: {step}")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = RealTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    step_count = 0

    if render:
        env.render_env(taxi_pos=env.taxi_loc,
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)

    while not done and not truncated:
        action = student_agent.get_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1

        if render:
            env.render_env(taxi_pos=env.taxi_loc,
                           action=action, step=step_count, fuel=env.current_fuel)
            time.sleep(0.5)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")