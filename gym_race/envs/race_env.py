import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D

class RaceEnv(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}
    def __init__(self, render_mode="human", continuous_state=False, continuous_action=False):
        print("init RaceEnv")
        # Configure action space
        if continuous_action:
            # Continuous action space: [acceleration, steering]
            # acceleration: -1 (brake) to 1 (accelerate)
            # steering: -1 (turn right) to 1 (turn left)
            self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            # Discrete actions: [accelerate, turn left, turn right, brake]
            self.action_space = spaces.Discrete(4)  # Added brake action
        
        # Configure observation space
        if continuous_state:
            # Continuous state space: 5 radar readings (normalized distances)
            self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        else:
            # Discrete state space (as in original)
            self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]), high=np.array([10, 10, 10, 10, 10]), dtype=int)
        
        self.continuous_state = continuous_state
        self.continuous_action = continuous_action
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view, continuous_radar=continuous_state)
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs=[]
        self.pyrace = PyRace2D(self.is_view, mode=self.render_mode, continuous_radar=self.continuous_state)
        obs = self.pyrace.observe()
        return np.array(obs, dtype=np.float32 if self.continuous_state else int), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done = self.pyrace.is_done()
        obs = self.pyrace.observe()
        info = {
            'dist': self.pyrace.car.distance, 
            'check': self.pyrace.car.current_check, 
            'crash': not self.pyrace.car.is_alive,
            'speed': self.pyrace.car.speed
        }
        return np.array(obs, dtype=np.float32 if self.continuous_state else int), reward, done, False, info

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self): # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
