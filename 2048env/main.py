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
        """
        Execute one step in the environment
        
        Returns:
            observation: current board state (copy to prevent external modification)
            reward: sum of merged tiles this step
            done: whether the game is over
            truncated: always False
            info: empty dict
        """
        old_board = self.board.copy()
        self.merge_reward = 0
        
        self._move(action)
        
        if np.array_equal(old_board, self.board):
            reward = 0
        else:
            self._spawn()
            reward = self.merge_reward
            self.score += reward
        
        done = self._is_done()
        
        return self.board.copy(), reward, done, False, {}

    def _move(self, action):
        """Execute the move based on action (0=up, 1=down, 2=left, 3=right)"""
        if action == 0:  # Up
            self.board = self._move_up()
        elif action == 1:  # Down
            self.board = self._move_down()
        elif action == 2:  # Left
            self.board = self._move_left()
        elif action == 3:  # Right
            self.board = self._move_right()
        
    def _move_up(self):
        return

    def _move_down(self):
        return
    
    def _move_left(self):
        return
    
    def _move_right(self):
        return

    def _is_done(self):
        """
        Check if the game is over
        Game ends when there are no empty cells and no possible merges
        """
        if np.any(self.board == 0):
            return False
        
        if np.any(self.board[:, :-1] == self.board[:, 1:]):
            return False
        
        if np.any(self.board[:-1, :] == self.board[1:, :]):
            return False
        
        return True


    def render(self):
        """Print the current board state"""
        print("\n" + "="*25)
        for row in self.board:
            print("|" + "|".join(f"{val:4d}" if val != 0 else "    " for val in row) + "|")
            print("-"*25)
        print(f"Score: {self.score}")
        print("="*25)

def human_mode():
    """Allows human to play the game with arrow keys"""
    env = Game2048Env()
    return
    

def auto_mode():
    """Test the 2048 environment with random actions"""
    env = Game2048Env()
    
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    print("Starting 2048 Game - Auto Mode!")
    print("Initial board:")
    env.render()

    valid_moves = 0
    max_attempts = 4

    for _ in range(1, max_attempts + 1):
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)

        if reward > 0:
            valid_moves += 1
            print(f"\nMove {valid_moves} - Action: {action_names[action]}")
            env.render()
            print(f"Reward: {reward} | Score: {env.score} | Max: {obs.max()}")
        
        if done:
            print("\n" + "="*50)
            print("Game Over!")
            print(f"Final Score: {env.score}")
            print(f"Max Tile: {env.board.max()}")
            print("="*50)
            break
    
    if not done:
        print("\nMax attempts reached. Ending game.")
        print(f"Final Score: {env.score}")
        print(f"Max Tile: {env.board.max()}")    
    


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
