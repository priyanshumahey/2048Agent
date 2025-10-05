import sys
import tty
import termios
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import jit

@jit(nopython=True)
def _move_left_jit(board):
    """Numba-optimized left move with merging logic"""
    new_board = np.zeros((4, 4), dtype=np.int32)
    merge_reward = 0
    
    for i in range(4):
        row = board[i][board[i] != 0]

        if len(row) == 0:
            continue
        
        merged = []
        j = 0
        while j < len(row):
            if j + 1 < len(row) and row[j] == row[j + 1]:
                merged_value = row[j] * 2
                merged.append(merged_value)
                merge_reward += merged_value
                j += 2
            else:
                merged.append(row[j])
                j += 1

        for k, val in enumerate(merged):
            new_board[i][k] = val
    
    return new_board, merge_reward

@jit(nopython=True)
def _is_done_jit(board):
    """Numba-optimized game over check"""
    if np.any(board == 0):
        return False
    
    for i in range(4):
        for j in range(3):
            if board[i, j] == board[i, j + 1]:
                return False
    
    for i in range(3):
        for j in range(4):
            if board[i, j] == board[i + 1, j]:
                return False
    
    return True


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
        self.observation_space = spaces.Box(low=0, high=131072, shape=(4, 4), dtype=np.int32)
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
            info: dict with 'valid_move' key indicating if the move changed the board
        """
        old_board = self.board.copy()
        self.merge_reward = 0
        
        self._move(action)
        
        valid_move = not np.array_equal(old_board, self.board)
        
        if valid_move:
            self._spawn()
            reward = self.merge_reward
            self.score += reward
        else:
            reward = 0
        
        done = self._is_done()
        
        return self.board.copy(), reward, done, False, {'valid_move': valid_move}

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
        
    def _move_left(self):
        """Move all tiles left and merge"""
        new_board, merge_reward = _move_left_jit(self.board)
        self.merge_reward += merge_reward
        return new_board

    def _move_right(self):
        """Move all tiles right and merge"""
        flipped = np.fliplr(self.board)
        new_board, merge_reward = _move_left_jit(flipped)
        self.merge_reward += merge_reward
        return np.fliplr(new_board)

    def _move_up(self):
        """Move all tiles up and merge"""
        transposed = self.board.T
        new_board, merge_reward = _move_left_jit(transposed)
        self.merge_reward += merge_reward
        return new_board.T

    def _move_down(self):
        """Move all tiles down and merge"""
        transposed = self.board.T
        flipped = np.fliplr(transposed)
        new_board, merge_reward = _move_left_jit(flipped)
        self.merge_reward += merge_reward
        result = np.fliplr(new_board)
        return result.T

    def _is_done(self):
        """
        Check if the game is over
        Game ends when there are no empty cells and no possible merges
        """
        return _is_done_jit(self.board)


    def render(self):
        """Print the current board state"""
        print("\n" + "="*25)
        for row in self.board:
            print("|" + "|".join(f"{val:4d}" if val != 0 else "    " for val in row) + "|")
            print("-"*25)
        print(f"Score: {self.score}")
        print("="*25)
    
def get_arrow_key():
    """Get arrow key input from terminal"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        
        if ch == '\x1b':
            ch = sys.stdin.read(2)
            if ch == '[A':
                return 0
            elif ch == '[B':
                return 1
            elif ch == '[D':
                return 2
            elif ch == '[C':
                return 3
        elif ch == 'q':
            return -1
        elif ch == 'r':
            return -2
            
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def human_mode():
    """Allows human to play the game with arrow keys"""
    env = Game2048Env()

    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    print("Starting 2048 Game - Human Mode")
    print("Use arrow keys to move tiles. Press 'q' to quit, 'r' to restart.")
    print("Press any key to start...")
    get_arrow_key()

    while True:
        print("\033[2J\033[H", end="")

        env.render()

        if env._is_done():
            print("Game Over!")
            print(f"Final Score: {env.score}")
            print(f"Max Tile: {env.board.max()}")

            break
        
        action = get_arrow_key()
        
        if action == -1:
            print("\n👋 Thanks for playing!")
            break
        elif action == -2:
            env.reset()
            print("\n🔄 Game restarted!")
            continue
        elif action is None:
            continue
        
        obs, reward, done, truncated, info = env.step(action)
        
        if info['valid_move']:
            if reward > 0:
                print(f"\n✓ {action_names[action]} - Merged tiles! +{reward} points")
            else:
                print(f"\n✓ {action_names[action]} - Tiles moved")
        else:
            print(f"\n✗ Invalid move! No tiles can move {action_names[action]}. Try a different direction.")

def auto_mode():
    """Test the 2048 environment with random actions"""
    env = Game2048Env()
    
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    print("Starting 2048 Game - Auto Mode!")
    print("Initial board:")
    env.render()

    move_count = 0
    max_moves = 100

    for _ in range(max_moves):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        if info['valid_move']:
            move_count += 1
            print(f"\nMove {move_count} - Action: {action_names[action]}")
            env.render()
            if reward > 0:
                print(f"Reward: {reward} | Score: {env.score} | Max: {obs.max()}")
            else:
                print(f"Tiles moved (no merge) | Score: {env.score} | Max: {obs.max()}")
        
        if done:
            print("\n" + "="*50)
            print("Game Over!")
            print(f"Final Score: {env.score}")
            print(f"Max Tile: {env.board.max()}")
            print(f"Total Valid Moves: {move_count}")
            print("="*50)
            break
    
    if not done:
        print("\nMax moves reached. Ending game.")
        print(f"Final Score: {env.score}")
        print(f"Max Tile: {env.board.max()}")
        print(f"Total Valid Moves: {move_count}")    

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
