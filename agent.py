# Reinforced learning - Agent for SnakeAI.py

import random
import snakeAI 
from snakeAI import SnakegameApp
import time
import numpy as np
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent: 

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0   # Control randomness
        self.gamma   = 0   # Discount rate
        self.memory  = deque(maxlen=MAX_MEMORY)  # popleft() if full

        self.game = SnakegameApp(game_engine_hook=self.game_hook, running_ai=True)
        self.game.run()

        # TODO: model, trainer
        pass
    
    def game_engine(self):
        pass



    def get_state(self): 
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores     = []
    plot_mean_sores = []
    total_score     = 0
    record          = 0
    agent = Agent()
    game = SnakegameApp(game_engine_hook=agent.game_engine, running_ai=True)
    game.run()
    pass

if __name__ == '__main__':
    train()
