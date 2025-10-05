import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Game2048Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.board.fill(0)
        self._spawn()
        self._spawn()
        return self.board.copy(), {}

    def _spawn(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            i, j = empty[np.random.randint(len(empty))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4

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


def main():
    env = Game2048Env()
    print("Initial board:")
    env.render()
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"\nAfter action {action}:")
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        
        if done:
            break

if __name__ == "__main__":
    main()
