import numpy as np
import gymnasium as gym
from gymnasium import spaces
import argparse

class Game2048Env(gym.Env):
    """
    2048 Game Environment following Gymnasium API
    
    Action space: Discrete(4)
        0: Up
        1: Down
        2: Left
        3: Right
    
    Observation space: Box(0, 2048, (4, 4), int32)
        4x4 grid of tile values
    """

    def __init__(self):
        super().__init__()
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)
        self.score = 0
        self.merge_reward = 0
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state with two tiles"""
        if seed is not None:
            np.random.seed(seed)
        self.board.fill(0)
        self.score = 0
        self.merge_reward = 0
        self._spawn(value=2)
        self._spawn(value=2)
        return self.board.copy(), {}

    def _spawn(self, value=None):
        """Spawn a new tile in a random empty cell"""
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            i, j = empty[np.random.randint(len(empty))]
            if value is None:
                self.board[i, j] = 2
            else:
                self.board[i, j] = value

    def step(self, action):
        old_board = self.board.copy()
        self._move(action)
        if not np.array_equal(old_board, self.board):
            self._spawn()
        done = self._is_done()
        reward = self.board.max()
        return self.board.copy(), reward, done, False, {}

    def _move(self, action):
        pass

    def _is_done(self):
        return False

    def render(self):
        print(self.board)


def human_mode():
    return

def auto_mode():
    return

def main():
    parser = argparse.ArgumentParser(description='2048 Game Environment')
    parser.add_argument('--mode', type=str, default='auto', 
                       choices=['human', 'auto'],
                       help='Game mode: human (play with arrow keys) or auto (random moves)')
    
    args = parser.parse_args()
    
    if args.mode == 'human':
        human_mode()
    else:
        auto_mode()

if __name__ == "__main__":
    main()
