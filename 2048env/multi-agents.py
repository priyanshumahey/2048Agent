import ray
import numpy as np
from main import Game2048Env

ray.init()

@ray.remote
class EnvWorker:
    def __init__(self):
        self.env = Game2048Env()

    def run_episode(self, policy_fn):
        obs, _ = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, done, _, _ = self.env.step(action)
            total_reward += reward
        return total_reward

workers = [EnvWorker.remote() for _ in range(8)]
 
def random_policy(obs): return np.random.randint(4)

results = ray.get([w.run_episode.remote(random_policy) for w in workers])
print(results)
 