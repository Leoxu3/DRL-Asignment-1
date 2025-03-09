import random

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
        self.obstacles = set(random.sample(list(all_locations), random.randint(0, self.grid_size)))
        all_locations -= self.obstacles
        self.taxi_loc = random.choice(list(all_locations))
        self.stations = random.sample(list(all_locations), 4)
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
        elif action == 4:  
            if (self.taxi_loc == self.passenger_loc) and (not self.passenger_picked_up):
                reward += 10
                self.passenger_picked_up = True
            else:
                reward -= 10
        elif action == 5:
            if (self.taxi_loc == self.destination) and self.passenger_picked_up:
                reward += 50
                done = True
            else:
                reward -= 10
                self.passenger_picked_up = False
        
        if self.current_fuel <= 0:
            done = True
            reward -= 10

        return self.get_state(), reward, done, False, {}
