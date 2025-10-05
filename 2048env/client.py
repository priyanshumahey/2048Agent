import asyncio
import websockets
import json
import numpy as np
import argparse
from typing import Optional

class Game2048Client:
    """Client that connects to 2048 game server and plays using different strategies"""
    
    def __init__(self, server_url: str = "ws://localhost:8000", session_id: str = "agent1"):
        self.server_url = f"{server_url}/ws/{session_id}"
        self.session_id = session_id
        self.game_count = 0
        self.total_score = 0
        self.max_tile_achieved = 0
    
    async def random_agent(self, max_games: Optional[int] = None, delay: float = 0.1):
        """
        Random agent - picks random valid moves
        
        Args:
            max_games: Number of games to play (None = infinite)
            delay: Delay be tween moves in seconds
        """
        print(f"Connecting to: {self.server_url}")
        
        async with websockets.connect(self.server_url) as ws:
            move_count = 0
            
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                if data["type"] == "init":
                    print(f"\n Game {self.game_count + 1} Started!")
                    print(f"Initial state received")
                    move_count = 0
                    
                elif data["type"] == "reset":
                    self.game_count += 1
                    print(f"\n Game {self.game_count} Stats:")
                    print(f"   Final Score: {self.total_score}")
                    print(f"   Max Tile: {self.max_tile_achieved}")
                    print(f"   Total Moves: {move_count}")

                    if max_games and self.game_count >= max_games:
                        print(f"\n Completed {max_games} games. Disconnecting...")
                        break
                    
                    move_count = 0
                    self.total_score = 0
                    self.max_tile_achieved = 0
                    continue
                    
                elif data["type"] == "step":
                    state = np.array(data["state"])
                    score = data["score"]
                    reward = data["reward"]
                    done = data["done"]
                    valid_move = data["valid_move"]
                    max_tile = data["max_tile"]
                    
                    if valid_move:
                        move_count += 1
                        self.total_score = score
                        self.max_tile_achieved = max(self.max_tile_achieved, max_tile)
                        
                        if reward > 0:
                            print(f"Move {move_count}: {data['action_name']} ✓ | "
                                  f"Reward: +{reward} | Score: {score} | Max: {max_tile}")
                    
                    if done:
                        print(f"\n Game Over!")
                        continue
                
                action = np.random.randint(4)
                await ws.send(json.dumps({"action": int(action)}))
                
                if delay > 0:
                    await asyncio.sleep(delay)
    
    async def corner_strategy_agent(self, max_games: Optional[int] = None, delay: float = 0.1):
        """
        Corner strategy - tries to keep high tiles in corners
        Priority: Left -> Down -> Up -> Right
        """
        print(f"Connecting to: {self.server_url}")
        
        async with websockets.connect(self.server_url) as ws:
            move_count = 0
            action_priority = [2, 1, 0, 3]
            last_valid_action = None
            
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                if data["type"] == "init":
                    print(f"\n Game {self.game_count + 1} Started!")
                    move_count = 0
                    last_valid_action = None
                    
                elif data["type"] == "reset":
                    self.game_count += 1
                    print(f"\n Game {self.game_count} Stats:")
                    print(f"   Final Score: {self.total_score}")
                    print(f"   Max Tile: {self.max_tile_achieved}")
                    print(f"   Total Moves: {move_count}")
                    
                    if max_games and self.game_count >= max_games:
                        print(f"\n Completed {max_games} games. Disconnecting...")
                        break
                    
                    move_count = 0
                    self.total_score = 0
                    self.max_tile_achieved = 0
                    last_valid_action = None
                    continue
                    
                elif data["type"] == "step":
                    valid_move = data["valid_move"]
                    
                    if valid_move:
                        move_count += 1
                        self.total_score = data["score"]
                        self.max_tile_achieved = max(self.max_tile_achieved, data["max_tile"])
                        last_valid_action = data["action"]
                        
                        if data["reward"] > 0:
                            print(f"Move {move_count}: {data['action_name']} ✓ | "
                                  f"Reward: +{data['reward']} | Score: {data['score']} | "
                                  f"Max: {data['max_tile']}")
                    
                    if data["done"]:
                        print(f"\n Game Over!")
                        continue
                
                if last_valid_action is not None and np.random.random() < 0.9:
                    action = last_valid_action
                else:
                    action = action_priority[move_count % len(action_priority)]
                
                await ws.send(json.dumps({"action": int(action)}))
                
                if delay > 0:
                    await asyncio.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description='2048 Game Websocket Client')
    parser.add_argument('--server', type=str, default='ws://localhost:8000',
                       help='WebSocket server URL (default: ws://localhost:8000)')
    parser.add_argument('--session', type=str, default='agent1',
                       help='Session ID for this client (default: agent1)')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'corner'],
                       help='Agent strategy: random or corner (default: random)')
    parser.add_argument('--games', type=int, default=None,
                       help='Number of games to play (default: infinite)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between moves in seconds (default: 0.1)')
    args = parser.parse_args()

    client = Game2048Client(server_url=args.server, session_id=args.session)

    try:
        if args.strategy == 'random':
            asyncio.run(client.random_agent(max_games=args.games, delay=args.delay))
        elif args.strategy == 'corner':
            asyncio.run(client.corner_strategy_agent(max_games=args.games, delay=args.delay))
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Disconnecting...")
    except Exception as e:
        print(f"\n Error: {e}")


if __name__ == "__main__":
    main()
